"""
Estimate M* per optimizer from E1 runs.

Primary estimator (proposal): for each M, paired over seeds, take
    delta(M) = Delta_ridge_new(bs=512) - Delta_ridge_new(bs=256).
Below M*, more sequences per task push the model toward dMMSE (farther from ridge on new
tasks): delta > 0. Above M*, toward ridge: delta < 0. M* = first + -> - crossing, linearly
interpolated in log2 M. 95% CI by bootstrapping seeds.

Single-run readout (for the controls): M where, on seen tasks, the model is equally far from
dMMSE and ridge, i.e. log(d_ridge_seen / d_dmmse_seen) crosses 0 (it is >0 below M*: closer to dMMSE).

  python analyze.py --runs runs [--opt adamw muon sgd]
"""
import argparse, glob, json
from collections import defaultdict
import numpy as np


def crossing(log2M, y):
    """First sign change of y from >0 to <=0, interpolated; nan if none."""
    for i in range(len(y) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] > 0 >= y[i + 1]:
            t = y[i] / (y[i] - y[i + 1])
            return log2M[i] + t * (log2M[i + 1] - log2M[i])
    return np.nan


def ci(samples):
    s = np.array(samples, float)
    s = s[np.isfinite(s)]
    return (np.percentile(s, 2.5), np.percentile(s, 97.5), len(s)) if len(s) else (np.nan,) * 2 + (0,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--opt", nargs="+", default=["adamw", "muon", "sgd"])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    R = {}  # (opt, M, bs, seed) -> final metrics
    for f in glob.glob(f"{a.runs}/*/result.json"):
        r = json.load(open(f)); c = r["config"]
        R[(c["opt"], c["M"], c["bs"], c["seed"])] = r["final"]

    rng = np.random.default_rng(0)
    for opt in a.opt:
        Ms = sorted({k[1] for k in R if k[0] == opt and k[1] > 0})
        seeds = sorted({k[3] for k in R if k[0] == opt})
        # paired delta[s, m]; nan if either run is missing
        delta = np.full((len(seeds), len(Ms)), np.nan)
        ratio = np.full((len(seeds), len(Ms)), np.nan)
        for si, s in enumerate(seeds):
            for mi, M in enumerate(Ms):
                lo, hi = R.get((opt, M, 256, s)), R.get((opt, M, 512, s))
                if lo and hi:
                    delta[si, mi] = hi["new"]["d_ridge"] - lo["new"]["d_ridge"]
                if lo:
                    ratio[si, mi] = np.log(lo["seen"]["d_ridge"] / lo["seen"]["d_dmmse"])
        log2M = np.log2(Ms)

        print(f"\n=== {opt}  ({len(seeds)} seeds) ===")
        print(f"{'log2 M':>7} {'delta_ridge(512-256)':>22} {'log(dR/dD) seen@256':>21}")
        for mi, M in enumerate(Ms):
            print(f"{log2M[mi]:7.1f} {np.nanmean(delta[:, mi]):22.5f} {np.nanmean(ratio[:, mi]):21.3f}")

        for name, mat in [("M*  (batch-size sign change)", delta), ("M*  (equal-distance readout)", ratio)]:
            point = crossing(log2M, np.nanmean(mat, 0))
            boots = []
            for _ in range(a.boot):
                idx = rng.integers(0, len(seeds), len(seeds))
                boots.append(crossing(log2M, np.nanmean(mat[idx], 0)))
            lo, hi, n = ci(boots)
            print(f"{name}: log2 M* = {point:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  ({n}/{a.boot} boots with a crossing)")


if __name__ == "__main__":
    main()
