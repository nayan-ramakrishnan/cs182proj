#!/usr/bin/env bash
# Unattended pipeline: (wait for an already-running sweep) -> sweep -> select -> E1 -> analyze.
# Safe to rerun at any point: finished runs are skipped, unfinished ones resume.
#   tmux new -d -s pipeline 'bash pipeline.sh'
# Env: DROP_SGD=auto (drop SGD if it misses ridge level in the sweep, per the proposal) | yes | no
set -uo pipefail
cd "$(dirname "$0")"
source ~/icl-venv/bin/activate
: "${RUNS_DIR:?set RUNS_DIR first}"
DROP_SGD="${DROP_SGD:-auto}"
LOG="$RUNS_DIR/pipeline.log"
mkdir -p jobs "$RUNS_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "pipeline start (DROP_SGD=$DROP_SGD)"
while tmux has-session -t sweep 2>/dev/null; do sleep 30; done   # don't double-launch the sweep

log "sweep: launching/finishing"
python jobs.py sweep > jobs/sweep.jsonl
python launch.py jobs/sweep.jsonl --per_gpu 4 >> "$LOG" 2>&1

log "selecting hyperparameters"
if python jobs.py select >> "$LOG" 2>&1; then HP=hparams.json; else HP=hparams_fallback.json; log "select failed; using $HP"; fi

OPTS=$(python - "$HP" "$DROP_SGD" <<'EOF'
import json, sys
hp, drop = json.load(open(sys.argv[1])), sys.argv[2]
opts = [o for o in ("sgd", "adamw", "muon") if o in hp]
if "sgd" in opts and (drop == "yes" or (drop == "auto" and not hp["sgd"].get("reaches_ridge_level", True))):
    opts.remove("sgd")
print(",".join(opts))
EOF
)
log "E1 optimizers: $OPTS  (hparams from $HP)"
python jobs.py e1 --hparams "$HP" --opts "$OPTS" > jobs/e1.jsonl
log "E1: $(wc -l < jobs/e1.jsonl) runs"
python launch.py jobs/e1.jsonl --per_gpu 7 >> "$LOG" 2>&1

log "analysis"
python analyze.py >> "$LOG" 2>&1 && log "DONE: see $RUNS_DIR/analysis/summary.md" || log "analysis failed"
