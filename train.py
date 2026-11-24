"""One training run: in-context linear regression on a fixed pool of M tasks.

Resumable: checkpoints go to <out>/ckpt.pkl; rerunning the same command continues from there,
reproducing exactly the run that would have happened without interruption (data for step t is
drawn from fold_in(data_key, t)). A finished run writes <out>/final.json and is skipped if rerun.

Logged at every eval point (to <out>/metrics.jsonl and W&B), all normalized by D:
  train_loss                  mean training MSE over the steps since the last eval
  mse_{seen,new}              model MSE vs the targets on seen-task / new-task prompts
  d_ridge_{seen,new}          Delta_ridge = mean (f - ridge)^2 / D
  d_dmmse_{seen,new}          Delta_dMMSE = mean (f - dMMSE)^2 / D
Reference values (ridge / dMMSE MSEs and their distance) are in <out>/refs.json.

Example:
  python train.py --opt adamw --M 2048 --seed 0 --lr 1e-3 --wd 1e-4 --out runs/test/adamw
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_default_matmul_precision", "highest")

from icl import data, model, optim  # noqa: E402

PROJECT = "cs182-icl-optimizers"
DEFAULT_ENTITY = "cs182-proj"


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--opt", required=True, choices=["sgd", "adamw", "muon"])
    p.add_argument("--M", type=int, required=True, help="task pool size; 0 = fresh task per sequence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--wd", type=float, default=0.0, help="per-step decoupled decay at peak LR (see icl/optim.py)")
    p.add_argument("--lr_aux", type=float, default=1e-3, help="Muon only: Adam LR for non-hidden params")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=100, help="steps per compiled call")
    p.add_argument("--eval_n", type=int, default=2048, help="prompts per eval set")
    p.add_argument("--eval_every", type=int, default=10_000)
    p.add_argument("--n_log_evals", type=int, default=60, help="extra log-spaced evals (for the clocks)")
    p.add_argument("--ckpt_every", type=int, default=50_000)
    p.add_argument("--out", required=True)
    p.add_argument("--group", default="misc")
    p.add_argument("--name", default=None)
    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--stop_after", type=int, default=0, help="testing: checkpoint and exit at this step")
    a = p.parse_args(argv)
    a.name = a.name or os.path.basename(os.path.normpath(a.out))
    assert a.steps % a.chunk == 0 and a.eval_every % a.chunk == 0 and a.ckpt_every % a.chunk == 0
    return a


def eval_steps(a):
    logs = np.logspace(np.log10(a.chunk), np.log10(a.steps), a.n_log_evals)
    s = {0, a.steps} | {int(np.ceil(v / a.chunk) * a.chunk) for v in logs}
    s |= set(range(a.eval_every, a.steps + 1, a.eval_every))
    return sorted(x for x in s if 0 <= x <= a.steps)


def save_atomic(path, obj, mode="wb"):
    tmp = path + ".tmp"
    with open(tmp, mode) as f:
        if "b" in mode:
            pickle.dump(obj, f)
        else:
            json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def init_wandb(a, refs):
    if not a.wandb:
        return None
    try:
        import wandb
        run_id = hashlib.md5(f"{a.group}/{a.name}".encode()).hexdigest()[:16]
        run = wandb.init(project=PROJECT, entity=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
                         group=a.group, name=a.name, id=run_id, resume="allow",
                         config={k: v for k, v in vars(a).items() if k not in ("out", "stop_after")},
                         dir=a.out, settings=wandb.Settings(console="off"))
        run.summary.update({f"ref/{k}": v for k, v in refs.items()})
        return run
    except Exception as e:  # never let logging kill a run
        print(f"[wandb] disabled: {e!r}", flush=True)
        return None


def run(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    final_path, ckpt_path = os.path.join(a.out, "final.json"), os.path.join(a.out, "ckpt.pkl")
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    if os.path.exists(final_path):
        print(f"{a.out}: already finished", flush=True)
        return
    D = model.D
    root = jax.random.PRNGKey(a.seed)
    pool = data.task_pool(a.seed, a.M)
    fresh = pool is None
    pool_arr = jnp.zeros((1, D)) if fresh else pool
    params = model.init_params(jax.random.fold_in(root, 1))
    tx = optim.build(a.opt, a.lr, a.wd, a.steps, params, warmup_frac=a.warmup_frac, clip=a.clip,
                     lr_aux=a.lr_aux)
    opt_state = tx.init(params)
    data_key = jax.random.fold_in(root, 2)

    # ---- fixed eval sets + reference predictions (same for every optimizer at this seed/M)
    ek_seen, ek_new = jax.random.split(jax.random.fold_in(root, 3))
    sets = {"new": data.sample(ek_new, a.eval_n, None)}
    if not fresh:
        sets["seen"] = data.sample(ek_seen, a.eval_n, pool)
    ref_preds, refs = {}, {}
    for s, (x, y) in sets.items():
        r = jax.jit(data.ridge_preds)(x, y)
        ref_preds[s] = {"ridge": r}
        refs[f"mse_ridge_{s}"] = float(jnp.mean((r - y) ** 2) / D)
        if not fresh:
            dm = jax.jit(data.dmmse_preds)(x, y, pool)
            ref_preds[s]["dmmse"] = dm
            refs[f"mse_dmmse_{s}"] = float(jnp.mean((dm - y) ** 2) / D)
            refs[f"d_ridge_dmmse_{s}"] = float(jnp.mean((dm - r) ** 2) / D)
    save_atomic(os.path.join(a.out, "refs.json"), refs, mode="w")

    def loss_fn(p, x, y):
        return jnp.mean((model.apply(p, x, y) - y) ** 2)

    @jax.jit
    def train_chunk(params, opt_state, start, pool_arr):
        def body(carry, i):
            p, s = carry
            x, y = data.sample(jax.random.fold_in(data_key, start + i), a.bs, None if fresh else pool_arr)
            l, g = jax.value_and_grad(loss_fn)(p, x, y)
            u, s = tx.update(g, s, p)
            return (optax.apply_updates(p, u), s), l
        (params, opt_state), ls = jax.lax.scan(body, (params, opt_state), jnp.arange(a.chunk))
        return params, opt_state, ls.mean()

    @jax.jit
    def evaluate(params):
        out = {}
        for s, (x, y) in sets.items():
            f = model.apply(params, x, y)
            out[f"mse_{s}"] = jnp.mean((f - y) ** 2) / D
            for ref, pr in ref_preds[s].items():
                out[f"d_{ref}_{s}"] = jnp.mean((f - pr) ** 2) / D
        return out

    # ---- resume
    step = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        params, opt_state, step = ck["params"], ck["opt_state"], ck["step"]
        print(f"resumed from step {step}", flush=True)
    if os.path.exists(metrics_path):  # drop records past the checkpoint (they will be recomputed)
        with open(metrics_path) as f:
            keep = [l for l in f if l.strip() and 0 < json.loads(l)["step"] <= step] if step else []
        with open(metrics_path, "w") as f:
            f.writelines(keep)
    wb = init_wandb(a, refs)
    print(f"run {a.name}: opt={a.opt} M={a.M} seed={a.seed} lr={a.lr} wd={a.wd} "
          f"params={model.count_params(params)} devices={jax.devices()}", flush=True)

    t0, step0, losses, diverged = time.time(), step, [], False
    targets = [t for t in eval_steps(a) if t > step or (step == 0 and t == 0)] or [step]
    for t in targets:
        while step < t:
            params, opt_state, l = train_chunk(params, opt_state, step, pool_arr)
            losses.append(l)
            step += a.chunk
        rec = {k: float(v) for k, v in evaluate(params).items()}
        rec["step"] = step
        rec["train_loss"] = float(jnp.mean(jnp.stack(losses)) / D) if losses else None
        losses = []
        elapsed = time.time() - t0
        rec["steps_per_sec"] = (step - step0) / elapsed if step > step0 else None
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if wb is not None:
            wb.log({k: v for k, v in rec.items() if v is not None and k != "step"}, step=step)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if not all(np.isfinite(v) for k, v in rec.items() if isinstance(v, float)):
            diverged = True
            break
        if step % a.ckpt_every == 0 or step == a.steps or step == a.stop_after:
            save_atomic(ckpt_path, {"params": jax.device_get(params),
                                    "opt_state": jax.device_get(opt_state), "step": step})
        if a.stop_after and step >= a.stop_after:
            print(f"stop_after reached at step {step}", flush=True)
            return

    final = {"config": vars(a), "refs": refs, "final": rec, "diverged": diverged,
             "steps_done": step, "n_params": model.count_params(params)}
    save_atomic(final_path, final, mode="w")
    if wb is not None:
        wb.summary.update({"diverged": diverged})
        wb.finish()
    print(f"done: {a.out} diverged={diverged}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1:])
