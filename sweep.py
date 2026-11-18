"""
Launch the E1 grid: M x seed x optimizer x batch size, one process per run,
spread over GPUs. Finished runs (result.json exists) are skipped, so it is safe to re-launch.

  python sweep.py --gpus 8 --per_gpu 1 --out runs
  python sweep.py --dry_run                      # just print the job list
  python sweep.py --lr_json '{"adamw":1e-3,"muon":2e-3,"sgd":3e-2}'
"""
import argparse, itertools, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

M_GRID = [round(2 ** e) for e in [9, 10, 10.5, 11, 11.5, 12, 12.5, 13, 15]]
OPTS = ["adamw", "muon", "sgd"]
BATCH_SIZES = [256, 512]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--opts", nargs="+", default=OPTS)
    ap.add_argument("--steps", type=int, default=500_000)  # identical for both batch sizes
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lr_json", default="{}")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    lrs = json.loads(a.lr_json)

    # seed-major order so that complete seeds finish first
    jobs = [(o, M, bs, s) for s, M, o, bs in
            itertools.product(range(a.seeds), M_GRID, a.opts, BATCH_SIZES)]
    jobs = [j for j in jobs if not os.path.exists(
        os.path.join(a.out, f"{j[0]}_M{j[1]}_bs{j[2]}_s{j[3]}", "result.json"))]
    print(f"{len(jobs)} runs to do")
    if a.dry_run:
        for j in jobs: print(j)
        return

    slots = Queue()
    for g in range(a.gpus):
        for _ in range(a.per_gpu):
            slots.put(g)

    def run(job):
        o, M, bs, s = job
        g = slots.get()
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                       XLA_PYTHON_CLIENT_MEM_FRACTION=str(round(0.9 / a.per_gpu, 2)))
            cmd = ["python", "train.py", "--opt", o, "--M", str(M), "--bs", str(bs), "--seed", str(s),
                   "--steps", str(a.steps), "--wd", str(a.wd), "--out", a.out]
            if o in lrs:
                cmd += ["--lr", str(lrs[o])]
            rc = subprocess.call(cmd, env=env)
            if rc != 0:
                print("FAILED", job)
        finally:
            slots.put(g)

    with ThreadPoolExecutor(a.gpus * a.per_gpu) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
