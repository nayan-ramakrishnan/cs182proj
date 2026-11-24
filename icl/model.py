"""Small GPT-2-style transformer for in-context linear regression, written in plain JAX.

Spec (project table): 4 layers, width 64, 2 heads, input = linear map from 9 numbers per token,
sequence length 32 (16 x/y pairs), 202,752 parameters. The read-in and read-out maps have no
bias, which is what makes the count come out to exactly 202,752.

Token layout (Raventos et al.): position 2k holds x_k as [0, x_k] and position 2k+1 holds y_k as
[y_k, 0, ..., 0]. The prediction of y_k is read at the x_k token, so it sees
(x_1, y_1, ..., x_{k-1}, y_{k-1}, x_k) and never y_k itself.
"""
import math

import jax
import jax.numpy as jnp

D = 8            # input dimension
K = 16           # in-context examples per sequence
D_IN = D + 1     # numbers per token
SEQ = 2 * K      # tokens per sequence
WIDTH = 64
LAYERS = 4
HEADS = 2
HIDDEN_MATRICES = ("q", "k", "v", "o", "fc1", "fc2")  # Muon is applied to these only


def _dense(key, n_in, n_out, std, bias=True):
    p = {"w": std * jax.random.normal(key, (n_in, n_out), jnp.float32)}
    if bias:
        p["b"] = jnp.zeros((n_out,), jnp.float32)
    return p


def _ln(n):
    return {"g": jnp.ones((n,), jnp.float32), "b": jnp.zeros((n,), jnp.float32)}


def init_params(key, width=WIDTH, layers=LAYERS):
    """GPT-2 init: N(0, 0.02), residual output projections scaled by 1/sqrt(2 * layers)."""
    std, proj_std = 0.02, 0.02 / math.sqrt(2 * layers)
    keys = jax.random.split(key, 3 + 6 * layers)
    params = {
        "readin": _dense(keys[0], D_IN, width, std, bias=False),
        "pos": std * jax.random.normal(keys[1], (SEQ, width), jnp.float32),
        "blocks": [],
        "ln_f": _ln(width),
        "readout": _dense(keys[2], width, 1, std, bias=False),
    }
    for l in range(layers):
        kq, kk, kv, ko, k1, k2 = keys[3 + 6 * l: 9 + 6 * l]
        params["blocks"].append({
            "ln1": _ln(width),
            "q": _dense(kq, width, width, std),
            "k": _dense(kk, width, width, std),
            "v": _dense(kv, width, width, std),
            "o": _dense(ko, width, width, proj_std),
            "ln2": _ln(width),
            "fc1": _dense(k1, width, 4 * width, std),
            "fc2": _dense(k2, 4 * width, width, proj_std),
        })
    return params


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def _layernorm(p, x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * p["g"] + p["b"]


def _lin(p, x):
    y = x @ p["w"]
    return y + p["b"] if "b" in p else y


def build_tokens(x, y):
    """x: (B, K, D), y: (B, K) -> tokens (B, 2K, D+1)."""
    B = x.shape[0]
    xt = jnp.concatenate([jnp.zeros((B, K, 1), x.dtype), x], -1)
    yt = jnp.concatenate([y[..., None], jnp.zeros((B, K, D), x.dtype)], -1)
    return jnp.stack([xt, yt], axis=2).reshape(B, SEQ, D_IN)


def apply(params, x, y):
    """Returns predictions of y_1..y_K, shape (B, K)."""
    h = _lin(params["readin"], build_tokens(x, y)) + params["pos"]
    B, T, W = h.shape
    hd = W // HEADS
    mask = jnp.tril(jnp.ones((T, T), bool))
    for blk in params["blocks"]:
        a = _layernorm(blk["ln1"], h)
        q = _lin(blk["q"], a).reshape(B, T, HEADS, hd)
        k = _lin(blk["k"], a).reshape(B, T, HEADS, hd)
        v = _lin(blk["v"], a).reshape(B, T, HEADS, hd)
        att = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(hd)
        att = jax.nn.softmax(jnp.where(mask, att, -1e30), axis=-1)
        o = jnp.einsum("bhts,bshd->bthd", att, v).reshape(B, T, W)
        h = h + _lin(blk["o"], o)
        a = _layernorm(blk["ln2"], h)
        h = h + _lin(blk["fc2"], jax.nn.gelu(_lin(blk["fc1"], a), approximate=True))
    h = _layernorm(params["ln_f"], h)
    out = _lin(params["readout"], h)[..., 0]  # (B, 2K)
    return out[:, 0::2]                       # read at the x_k tokens


def _names(path):
    return [getattr(p, "key", getattr(p, "idx", None)) for p in path]


def muon_labels(params):
    """'muon' for hidden weight matrices inside blocks, 'adam' for everything else."""
    def lab(path, _):
        n = _names(path)
        return "muon" if n[0] == "blocks" and n[-2] in HIDDEN_MATRICES and n[-1] == "w" else "adam"
    return jax.tree_util.tree_map_with_path(lab, params)


def decay_mask(params):
    """Weight decay on weight matrices only (not biases, LayerNorm gains, or position embeddings)."""
    return jax.tree_util.tree_map_with_path(lambda path, _: _names(path)[-1] == "w", params)
