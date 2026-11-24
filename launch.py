"""Self-healing job runner: packs runs onto GPUs and restarts crashed runs from their checkpoint.

  python launch.py jobs/e1.jsonl --per_gpu 6

Each line of the jobs file is {"group": ..., "name": ..., "args": {train.py flag: value}}.
Run outputs go to $RUNS_DIR/<group>/<name>/ (log.txt, metrics.jsonl, ckpt.pkl, final.json).
A job whose final.json exists is skipped, so rerunning the same command after anything goes
wrong (crash, reboot, Ctrl-C) just picks up where it left off. A job that exits without a
final.json is retried up to --retries times, resuming from its last checkpoint.
Progress is written to $RUNS_DIR/<jobs-file-stem>_status.json every few seconds.
Run it inside tmux so it survives disconnects.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))


def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return len([l for l in out.splitlines() if l.startswith("GPU")])
    except Exception:
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--root", default=os.environ.get("RUNS_DIR", os.path.join(HERE, "runs")))
    p.add_argument("--gpus", type=int, default=-1, help="-1 = all visible; 0 = CPU")
    p.add_argument("--per_gpu", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--wandb", type=int, default=1)
    a = p.parse_args()

    with open(a.jobs) as f:
        jobs = [json.loads(l) for l in f if l.strip()]
    gpus = n_gpus() if a.gpus < 0 else a.gpus
    slots = list(range(gpus)) if gpus > 0 else [None]
    cap = a.per_gpu
    status_path = os.path.join(a.root, os.path.splitext(os.path.basename(a.jobs))[0] + "_status.json")
    os.makedirs(a.root, exist_ok=True)

    def out_dir(j):
        return os.path.join(a.root, j["group"], j["name"])

    def done(j):
        return os.path.exists(os.path.join(out_dir(j), "final.json"))

    pending = deque(j for j in jobs if not done(j))
    finished = [j["name"] for j in jobs if done(j)]
    print(f"{len(jobs)} jobs, {len(finished)} already done, {len(pending)} to run on "
          f"{gpus or 'CPU'} x {cap} slots", flush=True)
    running, attempts, failed = {}, {}, []

    def shutdown(*_):
        for proc in running:
            proc.terminate()
        sys.exit(1)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    def write_status():
        st = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(jobs), "done": len(finished),
              "running": [r["job"]["name"] for r in running.values()], "pending": len(pending),
              "failed": failed, "attempts": attempts}
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, status_path)

    while pending or running:
        load = {g: 0 for g in slots}
        for r in running.values():
            load[r["gpu"]] += 1
        while pending:
            g = min(slots, key=lambda s: load[s])
            if load[g] >= cap:
                break
            j = pending.popleft()
            od = out_dir(j)
            os.makedirs(od, exist_ok=True)
            cmd = [sys.executable, os.path.join(HERE, "train.py"), "--out", od, "--name", j["name"],
                   "--group", j["group"], "--wandb", str(a.wandb)]
            for k, v in j["args"].items():
                cmd += [f"--{k}", str(v)]
            env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", TF_CPP_MIN_LOG_LEVEL="2")
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            log = open(os.path.join(od, "log.txt"), "a")
            log.write(f"\n==== launch {time.strftime('%H:%M:%S')} gpu={g} attempt={attempts.get(j['name'], 0) + 1}\n")
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[proc] = {"job": j, "gpu": g, "log": log}
            load[g] += 1
        for proc in list(running):
            if proc.poll() is None:
                continue
            r = running.pop(proc)
            r["log"].close()
            j = r["job"]
            if done(j):
                finished.append(j["name"])
                print(f"[{time.strftime('%H:%M:%S')}] done {j['name']} ({len(finished)}/{len(jobs)})", flush=True)
            else:
                attempts[j["name"]] = attempts.get(j["name"], 0) + 1
                if attempts[j["name"]] < a.retries:
                    print(f"[{time.strftime('%H:%M:%S')}] {j['name']} exited rc={proc.returncode}; retrying", flush=True)
                    pending.append(j)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] {j['name']} FAILED after {a.retries} attempts", flush=True)
                    failed.append(j["name"])
        write_status()
        time.sleep(5)
    write_status()
    print(f"all finished: {len(finished)} done, {len(failed)} failed {failed}", flush=True)


if __name__ == "__main__":
    main()
