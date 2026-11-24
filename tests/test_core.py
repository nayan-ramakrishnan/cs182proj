"""Sanity tests (CPU is fine):  python tests/test_core.py"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

from icl import data, model  # noqa: E402
import train  # noqa: E402


def test_param_count():
    n = model.count_params(model.init_params(jax.random.PRNGKey(0)))
    assert n == 202_752, n


def test_ridge_matches_numpy():
    x, y = data.sample(jax.random.PRNGKey(1), 4, None)
    r = np.asarray(data.ridge_preds(x, y))
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    for b in range(4):
        for k in range(model.K):
            X, Y = x[b, :k], y[b, :k]
            w = np.linalg.solve(X.T @ X + data.SIGMA2 * np.eye(model.D), X.T @ Y)
            assert abs(w @ x[b, k] - r[b, k]) < 1e-4, (b, k)


def test_dmmse_matches_numpy():
    pool = data.task_pool(0, 5)
    x, y = data.sample(jax.random.PRNGKey(2), 8, pool)
    dm = np.asarray(data.dmmse_preds(x, y, pool, chunk=4))
    x, y, W = (np.asarray(v, np.float64) for v in (x, y, pool))
    for b in range(8):
        for k in range(model.K):
            ll = -((y[b, :k, None] - x[b, :k] @ W.T) ** 2).sum(0) / (2 * data.SIGMA2)
            p = np.exp(ll - ll.max())
            p /= p.sum()
            assert abs((p * (x[b, k] @ W.T)).sum() - dm[b, k]) < 1e-4, (b, k)
    one = data.task_pool(0, 1)
    x, y = data.sample(jax.random.PRNGKey(3), 4, one)
    assert np.allclose(data.dmmse_preds(x, y, one, chunk=4), jnp.einsum("bkd,d->bk", x, one[0]), atol=1e-5)


def test_pools_nested():
    assert np.array_equal(data.task_pool(3, 512), data.task_pool(3, 2048)[:512])


def test_causality():
    """Prediction k must not depend on y_k or anything later."""
    p = model.init_params(jax.random.PRNGKey(0))
    x, y = data.sample(jax.random.PRNGKey(4), 2, None)
    base = model.apply(p, x, y)
    for k in (0, 5, 15):
        y2 = y.at[:, k:].add(10.0)
        x2 = x.at[:, k + 1:].add(10.0)
        assert np.allclose(model.apply(p, x, y2)[:, :k + 1], base[:, :k + 1], atol=1e-5), k
        assert np.allclose(model.apply(p, x2, y)[:, :k + 1], base[:, :k + 1], atol=1e-5), k


def _run(tmp, name, opt, extra=()):
    out = os.path.join(tmp, name)
    lr = {"sgd": 3e-2, "adamw": 3e-3, "muon": 2e-2}[opt]
    train.run(["--opt", opt, "--M", "16", "--lr", str(lr), "--wd", "1e-4", "--steps", "400", "--bs", "32",
               "--chunk", "50", "--eval_every", "100", "--n_log_evals", "3", "--ckpt_every", "100",
               "--eval_n", "64", "--wandb", "0", "--out", out, *extra])
    return out


def test_all_optimizers_train_and_resume():
    tmp = tempfile.mkdtemp()
    try:
        for opt in ("sgd", "adamw", "muon"):
            full = _run(tmp, f"{opt}_full", opt)
            with open(os.path.join(full, "final.json")) as f:
                fin = json.load(f)
            with open(os.path.join(full, "metrics.jsonl")) as f:
                curve = [json.loads(l) for l in f]
            assert not fin["diverged"], opt
            assert curve[-1]["mse_seen"] < curve[0]["mse_seen"], (opt, curve[0], curve[-1])
            part = _run(tmp, f"{opt}_part", opt, ["--stop_after", "200"])
            assert not os.path.exists(os.path.join(part, "final.json"))
            _run(tmp, f"{opt}_part", opt)  # resume
            with open(os.path.join(part, "final.json")) as f:
                fin2 = json.load(f)
            for k in ("mse_seen", "mse_new", "d_ridge_seen", "d_dmmse_seen"):
                assert abs(fin["final"][k] - fin2["final"][k]) < 1e-6, (opt, k, fin["final"][k], fin2["final"][k])
            print(f"  {opt}: mse_seen {curve[0]['mse_seen']:.3f} -> {curve[-1]['mse_seen']:.3f}; resume exact")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}", flush=True)
