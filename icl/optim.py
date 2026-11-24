"""SGD (momentum), AdamW, and Muon, sharing schedule, clipping and weight-decay conventions.

All three use: global-norm gradient clipping, linear warmup then cosine decay to 0, and
decoupled weight decay on weight matrices written as a per-step shrink factor:

    p <- p - omega * (lr_t / lr_peak) * p

so `omega` means the same thing for every optimizer regardless of its learning-rate scale.
omega = 1e-4 equals standard AdamW with lr=1e-3 and weight_decay=0.1.

Muon (Jordan et al.): Nesterov momentum (beta=0.95), then 5 Newton-Schulz steps to the nearest
semi-orthogonal matrix, scaled by sqrt(max(1, fan_out / fan_in)). Applied to hidden weight
matrices only (attention q/k/v/o and MLP); read-in, read-out, position embedding, LayerNorm and
biases are updated by Adam with learning rate `lr_aux`.
"""
import math

import jax
import jax.numpy as jnp
import optax

from . import model


def warmup_cosine(peak, steps, warmup_frac):
    warm = max(1, int(steps * warmup_frac))
    return optax.warmup_cosine_decay_schedule(0.0, peak, warm, steps, end_value=0.0)


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (jnp.linalg.norm(G) + 1e-7)
    transpose = G.shape[0] > G.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transpose else X


def scale_by_muon(beta=0.95, ns_steps=5):
    def init(params):
        return {"mu": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def update(g, state, params=None):
        mu = jax.tree_util.tree_map(lambda m, x: beta * m + (1 - beta) * x, state["mu"], g)
        u = jax.tree_util.tree_map(lambda m, x: (1 - beta) * x + beta * m, mu, g)  # Nesterov

        def orth(x):  # x is a (fan_in, fan_out) kernel
            return newton_schulz(x, ns_steps) * math.sqrt(max(1.0, x.shape[1] / x.shape[0]))
        return jax.tree_util.tree_map(orth, u), {"mu": mu}

    return optax.GradientTransformation(init, update)


def decoupled_decay(omega, frac_fn, mask):
    def init(params):
        return {"count": jnp.zeros([], jnp.int32)}

    def update(updates, state, params):
        f = omega * frac_fn(state["count"])
        updates = jax.tree_util.tree_map(lambda u, p, m: u - f * p if m else u, updates, params, mask)
        return updates, {"count": state["count"] + 1}

    return optax.GradientTransformation(init, update)


def build(name, lr, wd, steps, params, warmup_frac=0.05, clip=1.0, lr_aux=1e-3,
          momentum=0.9, b1=0.9, b2=0.999):
    sched = warmup_cosine(lr, steps, warmup_frac)
    if name == "adamw":
        core = optax.adam(sched, b1=b1, b2=b2, eps=1e-8)
    elif name == "sgd":
        core = optax.sgd(sched, momentum=momentum)
    elif name == "muon":
        core = optax.multi_transform(
            {"muon": optax.chain(scale_by_muon(), optax.scale_by_learning_rate(sched)),
             "adam": optax.adam(warmup_cosine(lr_aux, steps, warmup_frac), b1=b1, b2=b2, eps=1e-8)},
            model.muon_labels(params))
    else:
        raise ValueError(name)
    parts = [optax.clip_by_global_norm(clip)] if clip and clip > 0 else []
    parts.append(core)
    if wd > 0:
        parts.append(decoupled_decay(wd, lambda c: sched(c) / lr, model.decay_mask(params)))
    return optax.chain(*parts)
