#!/usr/bin/env bash
# E1 extension to lower task diversity (M = 2^3 ... 2^8.5), same tuned hyperparameters, same group "e1",
# then re-run the analysis over the full grid.   tmux new -d -s low 'bash extend_low.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ~/icl-venv/bin/activate
: "${RUNS_DIR:?set RUNS_DIR first}"
LOG="$RUNS_DIR/pipeline.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
HP=hparams.json; [ -f "$HP" ] || HP=hparams_fallback.json
OPTS="${OPTS:-adamw,muon}"
mkdir -p jobs
python jobs.py e1 --hparams "$HP" --opts "$OPTS" --log2m 3 4 5 6 7 8 8.5 > jobs/e1low.jsonl
log "E1-low: $(wc -l < jobs/e1low.jsonl) runs ($OPTS, M=2^3..2^8.5, hparams $HP)"
python launch.py jobs/e1low.jsonl --per_gpu 4 >> "$LOG" 2>&1
log "analysis"
python analyze.py >> "$LOG" 2>&1 && log "DONE: see $RUNS_DIR/analysis/summary.md" || log "analysis failed"
