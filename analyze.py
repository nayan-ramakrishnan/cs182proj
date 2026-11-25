"""E1 analysis: task-diversity threshold M* per optimizer and seed, plus the two clocks.

  python analyze.py            # reads $RUNS_DIR/e1, writes $RUNS_DIR/analysis/

M* readout (proposal): the smallest M at which Delta_ridge <= Delta_dMMSE at the end of training.
We report it two ways, on seen-task prompts (primary) and on new-task prompts:
  * grid:   the smallest grid M with Delta_ridge <= Delta_dMMSE
  * interp: crossing of log(Delta_ridge / Delta_dMMSE) = 0, linear in log2 M between grid points
A shift between optimizers "counts" only if it exceeds the spread across seeds.

Clocks (for the race prediction of M*):
  * t_icl: first eval step where new-task MSE is within --eps_icl of ridge's (learning clock)
  * t_mem: first eval step where, on seen tasks, Delta_dMMSE / (Delta_dMMSE + Delta_ridge)
           <= --r_mem (memorization clock)
  Predicted M*: fit log t_mem = a + b log M over runs that memorize, take T_icl = median t_icl
  of the largest-M runs, and solve t_mem(M) = T_icl.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = ["sgd", "adamw", "muon"]


def load(root):
    runs = []
    for fp in glob.glob(os.path.join(root, "e1", "*", "metrics.jsonl")):
        d = os.path.dirname(fp)
        if not os.path.exists(os.path.join(d, "final.json")):
            continue
        with open(os.path.join(d, "final.json")) as f:
            fin = json.load(f)
        with open(fp) as f:
            curve = [json.loads(l) for l in f if l.strip()]
        c = fin["config"]
        runs.append({"opt": c["opt"], "M": c["M"], "l2": math.log2(c["M"]), "seed": c["seed"],
                     "final": fin["final"], "refs": fin["refs"], "curve": curve, "diverged": fin["diverged"]})
    return runs


def mstar(l2s, dr, dd):
    """Return (grid M* as log2, interpolated M* as log2) or None/inf-style strings."""
    order = np.argsort(l2s)
    l2s, g = np.asarray(l2s)[order], np.log(np.asarray(dr)[order] / np.asarray(dd)[order])
    grid = next((l for l, gi in zip(l2s, g) if gi <= 0), None)
    interp = None
    for i in range(len(g) - 1):
        if g[i] > 0 and g[i + 1] <= 0:
            interp = l2s[i] + (l2s[i + 1] - l2s[i]) * g[i] / (g[i] - g[i + 1])
            break
    if interp is None and len(g):
        interp = -math.inf if g[0] <= 0 else math.inf
    return grid, interp


def first_step(curve, cond):
    return next((r["step"] for r in curve if r["step"] > 0 and cond(r)), None)


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "<grid" if x < 0 else ">grid"
    return f"2^{x:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("RUNS_DIR", os.path.join(HERE, "runs")))
    p.add_argument("--eps_icl", type=float, default=0.10)
    p.add_argument("--r_mem", type=float, default=0.25)
    a = p.parse_args()
    out = os.path.join(a.root, "analysis")
    os.makedirs(out, exist_ok=True)
    runs = load(a.root)
    if not runs:
        raise SystemExit(f"no finished e1 runs under {a.root}/e1")
    by = defaultdict(list)
    for r in runs:
        by[(r["opt"], r["seed"])].append(r)

    # ---- M* per (opt, seed)
    res = {}
    for (opt, seed), rs in sorted(by.items()):
        rs = [r for r in rs if not r["diverged"]]
        entry = {"n_runs": len(rs)}
        for split in ("seen", "new"):
            g, i = mstar([r["l2"] for r in rs], [r["final"][f"d_ridge_{split}"] for r in rs],
                         [r["final"][f"d_dmmse_{split}"] for r in rs])
            entry[f"grid_{split}"], entry[f"interp_{split}"] = g, i
        res[f"{opt}/s{seed}"] = entry

    # ---- seed spread and pairwise comparison (seen-task, interpolated)
    per_opt = defaultdict(list)
    for key, e in res.items():
        v = e["interp_seen"]
        if v is not None and math.isfinite(v):
            per_opt[key.split("/")[0]].append(v)
    comp = []
    opts = [o for o in OPTS if any(r["opt"] == o for r in runs)]
    cmp_opts = [o for o in opts if per_opt[o]]
    for i, o1 in enumerate(cmp_opts):
        for o2 in cmp_opts[i + 1:]:
            m1, m2 = np.mean(per_opt[o1]), np.mean(per_opt[o2])
            spread = max(np.ptp(per_opt[o1]), np.ptp(per_opt[o2]))
            comp.append({"pair": f"{o2} - {o1}", "delta_log2M": m2 - m1, "seed_spread_log2M": spread,
                         "exceeds_seed_spread": bool(abs(m2 - m1) > spread)})

    # ---- clocks
    clocks = []
    for r in runs:
        if r["diverged"]:
            continue
        rr, cv = r["refs"], r["curve"]
        t_icl = first_step(cv, lambda m: m["mse_new"] <= (1 + a.eps_icl) * rr["mse_ridge_new"])
        t_mem = first_step(cv, lambda m: m["d_dmmse_seen"] / (m["d_dmmse_seen"] + m["d_ridge_seen"]) <= a.r_mem)
        clocks.append({"opt": r["opt"], "M": r["M"], "l2": r["l2"], "seed": r["seed"], "t_icl": t_icl, "t_mem": t_mem})
    pred = {}
    for o in opts:
        cs = [c for c in clocks if c["opt"] == o]
        top = max(c["l2"] for c in cs)
        ticl = [c["t_icl"] for c in cs if c["l2"] == top and c["t_icl"]]
        mem = [c for c in cs if c["t_mem"]]
        entry = {"T_icl": float(np.median(ticl)) if ticl else None, "n_mem_points": len(mem)}
        if entry["T_icl"] and len({c["M"] for c in mem}) >= 2:
            b, a0 = np.polyfit(np.log([c["M"] for c in mem]), np.log([c["t_mem"] for c in mem]), 1)
            entry.update({"fit_slope": b, "fit_intercept": a0})
            if b > 0:
                entry["predicted_log2Mstar"] = (math.log(entry["T_icl"]) - a0) / b / math.log(2)
        pred[o] = entry

    # ---- write summary
    summary = {"mstar": res, "comparisons": comp, "clocks": clocks, "race_prediction": pred,
               "settings": vars(a)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    lines = ["# E1 summary", "", "## M* (end of training)", "",
             "| optimizer/seed | runs | seen: grid | seen: interp | new: grid | new: interp |",
             "|---|---|---|---|---|---|"]
    for k, e in res.items():
        lines.append(f"| {k} | {e['n_runs']} | {fmt(e['grid_seen'])} | {fmt(e['interp_seen'])} | "
                     f"{fmt(e['grid_new'])} | {fmt(e['interp_new'])} |")
    # memorization ratio r = Delta_dMMSE / (Delta_dMMSE + Delta_ridge) at end of training, per M
    # (r < 0.5: closer to dMMSE, i.e. memorizing; r > 0.5: closer to ridge)
    l2_all = sorted({r["l2"] for r in runs})
    lines += ["", "## End-of-training r = ΔdMMSE/(ΔdMMSE+Δridge) on seen tasks (mean over seeds; >0.5 = ridge-like)", "",
              "| optimizer | " + " | ".join(f"2^{l:g}" for l in l2_all) + " |",
              "|---|" + "---|" * len(l2_all)]
    for o in opts:
        cells = []
        for l in l2_all:
            vals = [r["final"]["d_dmmse_seen"] / (r["final"]["d_dmmse_seen"] + r["final"]["d_ridge_seen"])
                    for r in runs if r["opt"] == o and r["l2"] == l and not r["diverged"]]
            cells.append(f"{np.mean(vals):.2f}" if vals else "-")
        lines.append(f"| {o} | " + " | ".join(cells) + " |")
    # robustness: log2 M where r first rises past each threshold (linear in log2 M), seen tasks
    def r_cross(rs, t):
        pts = sorted((r["l2"], r["final"]["d_dmmse_seen"] / (r["final"]["d_dmmse_seen"] + r["final"]["d_ridge_seen"]))
                     for r in rs if not r["diverged"])
        for (l0, r0), (l1, r1) in zip(pts, pts[1:]):
            if r0 < t <= r1:
                return l0 + (l1 - l0) * (t - r0) / (r1 - r0)
        return None
    seeds = sorted({r["seed"] for r in runs})
    lines += ["", "## Crossover at other thresholds (seen tasks; log2 M where r first exceeds the threshold, per seed)", "",
              "| threshold | " + " | ".join(opts) + " |", "|---|" + "---|" * len(opts)]
    for t in (0.25, 0.5, 0.75):
        cells = []
        for o in opts:
            vals = [r_cross([r for r in runs if r["opt"] == o and r["seed"] == s], t) for s in seeds]
            cells.append(", ".join(f"{v:.2f}" if v is not None else "-" for v in vals))
        lines.append(f"| r = {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Optimizer differences vs seed spread (seen tasks, log2 M)", "",
              "| pair | delta | seed spread | exceeds? |", "|---|---|---|---|"]
    for c in comp:
        lines.append(f"| {c['pair']} | {c['delta_log2M']:+.2f} | {c['seed_spread_log2M']:.2f} | "
                     f"{'yes' if c['exceeds_seed_spread'] else 'no'} |")
    lines += ["", f"## Race prediction (eps_icl={a.eps_icl}, r_mem={a.r_mem})", "",
              "| optimizer | T_icl | memorizing runs | slope | predicted M* | measured M* (mean) |",
              "|---|---|---|---|---|---|"]
    for o, e in pred.items():
        meas = np.mean(per_opt[o]) if per_opt[o] else None
        lines.append(f"| {o} | {e['T_icl']} | {e['n_mem_points']} | "
                     f"{e.get('fit_slope', float('nan')):.2f} | {fmt(e.get('predicted_log2Mstar'))} | {fmt(meas)} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skipping plots")
        return
    colors = {"sgd": "C0", "adamw": "C1", "muon": "C2"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, split in zip(axes, ("seen", "new")):
        for r in runs:
            if r["diverged"]:
                continue
            c, mk = colors[r["opt"]], "o" if r["seed"] == 0 else "s"
            ax.scatter(r["l2"], r["final"][f"d_ridge_{split}"], color=c, marker=mk, facecolors="none")
            ax.scatter(r["l2"], r["final"][f"d_dmmse_{split}"], color=c, marker=mk)
        ax.set_yscale("log")
        ax.set_xlabel("log2 M")
        ax.set_title(f"{split} tasks: hollow = Δridge, filled = ΔdMMSE")
    for o, c in colors.items():
        axes[0].plot([], [], color=c, label=o)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "delta_vs_M.png"), dpi=130)

    fig, axes = plt.subplots(1, len(opts), figsize=(5 * len(opts), 4), squeeze=False)
    cmap = plt.get_cmap("viridis")
    l2s = sorted({r["l2"] for r in runs})
    for ax, o in zip(axes[0], opts):
        for r in sorted((r for r in runs if r["opt"] == o and r["seed"] == 0), key=lambda r: r["l2"]):
            cv = [m for m in r["curve"] if m["step"] > 0]
            ratio = [m["d_dmmse_seen"] / (m["d_dmmse_seen"] + m["d_ridge_seen"]) for m in cv]
            ax.plot([m["step"] for m in cv], ratio, color=cmap(l2s.index(r["l2"]) / max(1, len(l2s) - 1)),
                    label=f"2^{r['l2']:g}")
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(f"{o} (seed 0): ΔdMMSE/(ΔdMMSE+Δridge), seen")
        ax.set_xlabel("step")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "memorization_curves.png"), dpi=130)

    if not any(c["t_mem"] for c in clocks):
        print("no run reached the memorization criterion; skipping clocks.png")
        print(f"plots and summary written to {out}")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for o in opts:
        mem = [c for c in clocks if c["opt"] == o and c["t_mem"]]
        ax.scatter([c["M"] for c in mem], [c["t_mem"] for c in mem], color=colors[o], label=f"{o} t_mem")
        if pred[o]["T_icl"]:
            ax.axhline(pred[o]["T_icl"], color=colors[o], ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel("steps")
    ax.set_title("memorization clock vs M (dashed: T_icl)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "clocks.png"), dpi=130)
    print(f"plots and summary written to {out}")


if __name__ == "__main__":
    main()
