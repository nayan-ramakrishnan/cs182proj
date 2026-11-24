"""Compact progress view for a Terminal window:  watch -n 60 python status.py"""
import glob
import json
import os
import time

root = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
print(time.strftime("%H:%M:%S"), "RUNS_DIR =", root)
for fp in sorted(glob.glob(os.path.join(root, "*_status.json"))):
    try:
        s = json.load(open(fp))
    except Exception:
        continue
    name = os.path.basename(fp).replace("_status.json", "")
    print(f"{name:12} done {s['done']}/{s['total']}  running {len(s['running'])}  pending {s['pending']}  "
          f"failed {len(s['failed'])} {s['failed'] or ''}  (updated {s['time'][11:]})")
    group = "e1" if name == "e1" else name
    rows = []
    for r in s["running"][:60]:
        mp = os.path.join(root, group, r, "metrics.jsonl")
        try:
            last = json.loads(open(mp).read().strip().splitlines()[-1])
            rows.append(f"{r}:{last['step'] // 1000}k")
        except Exception:
            rows.append(f"{r}:-")
    for i in range(0, len(rows), 4):
        print("   " + "  ".join(f"{x:28}" for x in rows[i:i + 4]))
log = os.path.join(root, "pipeline.log")
if os.path.exists(log):
    print("--- pipeline.log")
    lines = [l.rstrip() for l in open(log) if l.startswith("[") or "selected" in l]
    print("\n".join(lines[-8:]))
summ = os.path.join(root, "analysis", "summary.md")
if os.path.exists(summ):
    print("--- analysis/summary.md")
    print(open(summ).read())
