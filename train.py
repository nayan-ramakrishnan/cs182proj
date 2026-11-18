"""
E1 single run: in-context linear regression with a fixed pool of M tasks.
Testbed of Raventos et al. 2023: D=8, K=16, sigma^2=0.25, 4-layer GPT-2-style, width 64.

Usage (sanity check):
  python train.py --opt adamw --M 2048 --bs 256 --steps 2000 --eval_every 1000 --out runs_test
M=0 means "a fresh task for every sequence" (M = n_train; needed later for E2).
"""
import argparse, json, os, time
from functools import partial

import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn
import flax.serialization
from flax import traverse_util
import optax

D, K, SIGMA2 = 8, 16, 0.25
SIGMA = SIGMA2 ** 0.5
LAMBDA = SIGMA2  # ridge strength sigma^2 / tau^2 with prior w ~ N(0, I)
WIDTH, LAYERS, HEADS = 64, 4, 8

# Placeholder learning rates: tune each optimizer separately (proposal: "own tuned LR").
DEFAULT_LR = {"adamw": 1e-3, "muon": 1e-3, "sgd": 1e-2}


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    @nn.compact
    def __call__(self, x):
        B, T, _ = x.shape
        h = nn.LayerNorm()(x)
        q, k, v = (nn.Dense(WIDTH, name=n)(h).reshape(B, T, HEADS, -1) for n in "qkv")
        att = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(WIDTH // HEADS)
        mask = jnp.tril(jnp.ones((T, T), bool))
        att = jax.nn.softmax(jnp.where(mask, att, -1e9), axis=-1)
        o = jnp.einsum("bhts,bshd->bthd", att, v).reshape(B, T, WIDTH)
        x = x + nn.Dense(WIDTH, name="o")(o)
        h = nn.LayerNorm()(x)
        h = nn.Dense(4 * WIDTH, name="fc1")(h)
        h = nn.Dense(WIDTH, name="fc2")(nn.gelu(h))
        return x + h


class ICLTransformer(nn.Module):
    @nn.compact
    def __call__(self, z):  # z: (B, 2K, D) interleaved x_1, y_1, x_2, y_2, ...
        T = z.shape[1]
        h = nn.Dense(WIDTH, name="readin")(z)
        h = h + self.param("pos", nn.initializers.normal(0.02), (T, WIDTH))
        for i in range(LAYERS):
            h = Block(name=f"block_{i}")(h)
        h = nn.LayerNorm(name="ln_f")(h)
        out = nn.Dense(1, name="readout")(h)[..., 0]  # (B, 2K)
        return out[:, 0::2]  # prediction of y_k read at the x_k token -> (B, K)


def build_tokens(xs, ys):
    ypad = jnp.concatenate([ys[..., None], jnp.zeros(ys.shape + (D - 1,))], -1)
    return jnp.stack([xs, ypad], axis=2).reshape(xs.shape[0], 2 * K, D)


# ----------------------------------------------------------------------------- data
def sample_batch(key, W, B):
    """W: (M, D) task pool, or None for a fresh task per sequence."""
    k1, k2, k3 = jax.random.split(key, 3)
    xs = jax.random.normal(k1, (B, K, D))
    if W is None:
        w = jax.random.normal(k2, (B, D))
    else:
        w = W[jax.random.randint(k2, (B,), 0, W.shape[0])]
    ys = jnp.einsum("bkd,bd->bk", xs, w) + SIGMA * jax.random.normal(k3, (B, K))
    return xs, ys


# ----------------------------------------------------------------------------- references
def ridge_preds(xs, ys):
    """Prediction for y_k from examples j<k (k=0 -> 0). Returns (B, K)."""
    outer = jnp.einsum("bkd,bke->bkde", xs, xs)
    A = LAMBDA * jnp.eye(D) + jnp.cumsum(outer, 1) - outer
    xy = ys[..., None] * xs
    b = jnp.cumsum(xy, 1) - xy
    w = jnp.linalg.solve(A, b[..., None])[..., 0]
    return jnp.einsum("bkd,bkd->bk", w, xs)


def dmmse_preds(xs, ys, W, chunk=32):
    """Bayes-optimal predictor for the discrete pool W (M, D). Returns (B, K)."""
    def one(args):
        x, y = args
        pw = jnp.einsum("bkd,md->bmk", x, W)
        se = (y[:, None, :] - pw) ** 2
        cum = jnp.cumsum(se, 2) - se                       # exclusive prefix sum over k
        post = jax.nn.softmax(-cum / (2 * SIGMA2), axis=1)  # posterior over pool
        what = jnp.einsum("bmk,md->bkd", post, W)
        return jnp.einsum("bkd,bkd->bk", what, x)
    B = xs.shape[0]
    out = jax.lax.map(one, (xs.reshape(B // chunk, chunk, K, D), ys.reshape(B // chunk, chunk, K)))
    return out.reshape(B, K)


# ----------------------------------------------------------------------------- optimizers
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(jnp.float32)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


def scale_by_muon(beta=0.95, nesterov=True):
    """Nesterov momentum -> nearest semi-orthogonal matrix. Rescaled by 0.2*sqrt(max dim)
    so the update RMS matches AdamW's and learning rates are roughly comparable."""
    def init(params):
        return optax.TraceState(trace=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update(grads, state, params=None):
        buf = jax.tree_util.tree_map(lambda g, m: beta * m + g, grads, state.trace)
        eff = jax.tree_util.tree_map(lambda g, m: g + beta * m, grads, buf) if nesterov else buf
        out = jax.tree_util.tree_map(lambda u: 0.2 * jnp.sqrt(max(u.shape)) * newton_schulz(u), eff)
        return out, optax.TraceState(trace=buf)
    return optax.GradientTransformation(init, update)


HIDDEN = {"q", "k", "v", "o", "fc1", "fc2"}


def muon_labels(params):
    flat = traverse_util.flatten_dict(params)
    lab = {p: ("muon" if p[-1] == "kernel" and p[-2] in HIDDEN else "adamw") for p in flat}
    return traverse_util.unflatten_dict(lab)


def make_optimizer(name, lr, wd, steps, clip):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup_steps=steps // 20, decay_steps=steps)
    if name == "adamw":
        core = optax.adamw(sched, weight_decay=wd)
    elif name == "sgd":
        core = optax.chain(optax.add_decayed_weights(wd), optax.sgd(sched, momentum=0.9, nesterov=True))
    elif name == "muon":
        core = optax.multi_transform(
            {"muon": optax.chain(scale_by_muon(), optax.add_decayed_weights(wd),
                                 optax.scale_by_learning_rate(sched)),
             "adamw": optax.adamw(sched, weight_decay=wd)},
            muon_labels)
    else:
        raise ValueError(name)
    return optax.chain(optax.clip_by_global_norm(clip), core) if clip > 0 else core


# ----------------------------------------------------------------------------- main
def train(a):
    run_dir = os.path.join(a.out, f"{a.opt}_M{a.M}_bs{a.bs}_s{a.seed}")
    if os.path.exists(os.path.join(run_dir, "result.json")):
        print("exists, skipping", run_dir); return
    os.makedirs(run_dir, exist_ok=True)

    lr = a.lr if a.lr is not None else DEFAULT_LR[a.opt]
    pool_key, init_key, data_key, eval_key = jax.random.split(jax.random.PRNGKey(a.seed), 4)
    W = jax.random.normal(pool_key, (a.M, D)) if a.M > 0 else None  # same pool for every optimizer

    model = ICLTransformer()
    params = model.init(init_key, jnp.zeros((1, 2 * K, D)))["params"]
    tx = make_optimizer(a.opt, lr, a.wd, a.steps, a.clip)
    opt_state = tx.init(params)

    def predict(p, xs, ys):
        return model.apply({"params": p}, build_tokens(xs, ys))

    def loss_fn(p, xs, ys):
        return jnp.mean((predict(p, xs, ys) - ys) ** 2)

    @partial(jax.jit, donate_argnums=(0, 1))
    def run_chunk(p, s, start):
        def body(carry, i):
            p, s = carry
            xs, ys = sample_batch(jax.random.fold_in(data_key, start + i), W, a.bs)
            loss, g = jax.value_and_grad(loss_fn)(p, xs, ys)
            upd, s = tx.update(g, s, p)
            return (optax.apply_updates(p, upd), s), loss
        (p, s), losses = jax.lax.scan(body, (p, s), jnp.arange(a.chunk))
        return p, s, losses.mean()

    @jax.jit
    def metrics(p, xs, ys, has_pool):
        f = predict(p, xs, ys)
        r = ridge_preds(xs, ys)
        m = {"loss": jnp.mean((f - ys) ** 2), "ridge_loss": jnp.mean((r - ys) ** 2),
             "d_ridge": jnp.mean((f - r) ** 2) / D}
        if has_pool:
            d = dmmse_preds(xs, ys, W)
            m.update(dmmse_loss=jnp.mean((d - ys) ** 2), d_dmmse=jnp.mean((f - d) ** 2) / D,
                     d_ridge_dmmse=jnp.mean((r - d) ** 2) / D)
        return {k: float(v) for k, v in m.items()}

    def evaluate(p, n):
        ks, kn = jax.random.split(eval_key)
        out = {}
        if W is not None:  # seen tasks, fresh inputs
            xs, ys = sample_batch(ks, W, n)
            out["seen"] = metrics(p, xs, ys, True)
        # new tasks: fresh w ~ N(0, I)
        xs, ys = sample_batch(kn, None, n)
        out["new"] = metrics(p, xs, ys, False)
        return out

    curve, t0 = [], time.time()
    for step in range(0, a.steps, a.chunk):
        params, opt_state, train_loss = run_chunk(params, opt_state, step)
        done = step + a.chunk
        if done % a.eval_every == 0 or done == a.steps:
            ev = evaluate(params, a.eval_n if done < a.steps else a.final_eval_n)
            curve.append({"step": done, "train_loss": float(train_loss), **ev})
            print(f"[{os.path.basename(run_dir)}] step {done} loss {float(train_loss):.4f} "
                  f"d_ridge(new) {ev['new']['d_ridge']:.4f} "
                  + (f"d_dmmse(seen) {ev['seen']['d_dmmse']:.4f} " if "seen" in ev else "")
                  + f"{time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(run_dir, "ckpt.msgpack"), "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    res = {"config": {**vars(a), "lr_used": lr, "D": D, "K": K, "sigma2": SIGMA2},
           "final": curve[-1], "curve": curve}
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--opt", choices=["adamw", "muon", "sgd"], required=True)
    p.add_argument("--M", type=int, required=True, help="task pool size; 0 = fresh task per sequence")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0, help="global-norm clip; 0 disables")
    p.add_argument("--chunk", type=int, default=1000, help="steps per jitted lax.scan")
    p.add_argument("--eval_every", type=int, default=10_000)
    p.add_argument("--eval_n", type=int, default=256)
    p.add_argument("--final_eval_n", type=int, default=2048)
    p.add_argument("--out", default="runs")
    train(p.parse_args())
