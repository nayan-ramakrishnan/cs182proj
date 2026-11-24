"""Task pools, on-the-fly sampling, and the two closed-form reference predictors (ridge, dMMSE)."""
import jax
import jax.numpy as jnp

from .model import D, K

SIGMA2 = 0.25          # label noise variance
POOL_MAX = 2 ** 15     # pools are nested: the pool for M is the first M rows of one 2^15 draw


def task_pool(seed, M):
    """M tasks w ~ N(0, I_D), fixed for the whole run. M == 0 means a fresh task per sequence."""
    if M == 0:
        return None
    assert 1 <= M <= POOL_MAX, M
    W = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(seed), 0), (POOL_MAX, D), jnp.float32)
    return W[:M]


def sample(key, B, pool):
    """Returns x (B, K, D), y (B, K). pool=None draws a fresh task per sequence."""
    k1, k2, k3 = jax.random.split(key, 3)
    if pool is None:
        w = jax.random.normal(k1, (B, D), jnp.float32)
    else:
        w = pool[jax.random.randint(k1, (B,), 0, pool.shape[0])]
    x = jax.random.normal(k2, (B, K, D), jnp.float32)
    y = jnp.einsum("bkd,bd->bk", x, w) + jnp.sqrt(SIGMA2) * jax.random.normal(k3, (B, K), jnp.float32)
    return x, y


def ridge_preds(x, y, lam=SIGMA2):
    """Posterior mean under w ~ N(0, I): predict y_k from (x_j, y_j)_{j<k}. Shape (B, K)."""
    outer = jnp.einsum("bkd,bke->bkde", x, x)
    xy = x * y[..., None]
    A = jnp.cumsum(outer, 1) - outer + lam * jnp.eye(D)   # exclusive prefix sums
    b = jnp.cumsum(xy, 1) - xy
    w = jnp.linalg.solve(A, b[..., None])[..., 0]          # (B, K, D)
    return jnp.einsum("bkd,bkd->bk", w, x)


def dmmse_preds(x, y, pool, chunk=64):
    """Bayes posterior mean under a uniform prior over the M pool tasks. Shape (B, K)."""
    def one(args):
        xc, yc = args
        P = jnp.einsum("bkd,md->bkm", xc, pool)            # every task's prediction
        r2 = (yc[..., None] - P) ** 2
        logits = -(jnp.cumsum(r2, 1) - r2) / (2 * SIGMA2)   # log-lik of the first k-1 pairs
        return (jax.nn.softmax(logits, -1) * P).sum(-1)
    B = x.shape[0]
    chunk = min(chunk, B)
    assert B % chunk == 0, (B, chunk)
    xs, ys = x.reshape(B // chunk, chunk, K, D), y.reshape(B // chunk, chunk, K)
    return jax.lax.map(one, (xs, ys)).reshape(B, K)
