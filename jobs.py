"""Generate job files for launch.py, and pick tuned hyperparameters from the sweep.

  python jobs.py bench  --copies 6 --tag p6 > jobs/bench_p6.jsonl   # speed test, 3 optimizers x copies
  python jobs.py sweep                    > jobs/sweep.jsonl      # LR x WD pilot, M = n_train, 50K steps
  python jobs.py select                                           # reads $RUNS_DIR/sweep, writes hparams.json
  python jobs.py e1 --hparams hparams.json > jobs/e1.jsonl        # 9 M x 2 seeds x 3 optimizers, 500K steps
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = ["sgd", "adamw", "muon"]
E1_LOG2M = [9, 10, 10.5, 11, 11.5, 12, 12.5, 13, 15]
LR_GRID = {
    "adamw": [3e-4, 1e-3, 3e-3, 1e-2],
    "muon": [5e-3, 1e-2, 2e-2, 5e-2],
    "sgd": [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
}
WD_GRID = [0.0, 1e-4]   # per-step decay at peak LR; 1e-4 == AdamW(lr=1e-3, weight_decay=0.1)
SWEEP_STEPS = 50_000
RIDGE_LEVEL = 0.05      # "reaches ridge" = new-task MSE within 5% of ridge's


def M_of(l2):
    return int(round(2 ** l2))


def emit(jobs):
    for j in jobs:
        print(json.dumps(j))


def cmd_bench(a):
    emit({"group": f"bench_{a.tag}", "name": f"{opt}_c{c}",
          "args": {"opt": opt, "M": 2048, "seed": c, "lr": {"sgd": 3e-2, "adamw": 1e-3, "muon": 2e-2}[opt],
                   "steps": a.steps, "eval_every": 1000, "n_log_evals": 5, "ckpt_every": a.steps}}
         for opt in OPTS for c in range(a.copies))


def cmd_sweep(a):
    emit({"group": "sweep", "name": f"{opt}_lr{lr:g}_wd{wd:g}",
          "args": {"opt": opt, "M": 0, "seed": 0, "lr": lr, "wd": wd, "steps": SWEEP_STEPS,
                   "eval_every": 5000, "n_log_evals": 30, "ckpt_every": 25_000}}
         for opt in OPTS for lr in LR_GRID[opt] for wd in WD_GRID)


def cmd_select(a):
    rows = []
    for fp in glob.glob(os.path.join(a.root, "sweep", "*", "final.json")):
        with open(fp) as f:
            r = json.load(f)
        c, fin = r["config"], r["final"]
        excess = math.inf if r["diverged"] else fin["mse_new"] / r["refs"]["mse_ridge_new"] - 1
        rows.append((c["opt"], c["lr"], c["wd"], excess))
    if not rows:
        sys.exit(f"no finished sweep runs under {a.root}/sweep")
    print(f"{'opt':6} {'lr':>8} {'wd':>7}  excess MSE over ridge (new tasks)")
    for o, lr, wd, ex in sorted(rows):
        print(f"{o:6} {lr:8g} {wd:7g}  {ex:.4f}")
    best = {}
    for o in OPTS:
        cand = [r for r in rows if r[0] == o]
        if cand:
            _, lr, wd, ex = min(cand, key=lambda r: r[3])
            best[o] = {"lr": lr, "wd": wd, "sweep_excess_over_ridge": ex,
                       "reaches_ridge_level": bool(ex <= RIDGE_LEVEL)}
    with open(a.out, "w") as f:
        json.dump(best, f, indent=1)
    print("\nselected:", json.dumps(best, indent=1))
    print(f"written to {a.out}")


def cmd_e1(a):
    with open(a.hparams) as f:
        hp = json.load(f)
    opts = a.opts.split(",")
    emit({"group": "e1", "name": f"{opt}_M2^{l2:g}_s{seed}",
          "args": {"opt": opt, "M": M_of(l2), "seed": seed, "lr": hp[opt]["lr"], "wd": hp[opt]["wd"],
                   "steps": a.steps}}
         for opt in opts for seed in a.seeds for l2 in (a.log2m or E1_LOG2M))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bench")
    b.add_argument("--copies", type=int, default=1)
    b.add_argument("--steps", type=int, default=5000)
    b.add_argument("--tag", default="p1")
    sp.add_parser("sweep")
    s = sp.add_parser("select")
    s.add_argument("--root", default=os.environ.get("RUNS_DIR", os.path.join(HERE, "runs")))
    s.add_argument("--out", default=os.path.join(HERE, "hparams.json"))
    e = sp.add_parser("e1")
    e.add_argument("--hparams", default=os.path.join(HERE, "hparams.json"))
    e.add_argument("--opts", default=",".join(OPTS))
    e.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    e.add_argument("--steps", type=int, default=500_000)
    e.add_argument("--log2m", type=float, nargs="+", default=None, help="override the M grid (log2 values)")
    a = p.parse_args()
    {"bench": cmd_bench, "sweep": cmd_sweep, "select": cmd_select, "e1": cmd_e1}[a.cmd](a)


if __name__ == "__main__":
    main()
