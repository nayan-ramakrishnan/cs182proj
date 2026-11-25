# How does the optimizer shape generalization in in-context regression?

CS182 project (Kaul, Liu, Ramakrishnan, Calderon). We measure the task diversity at which a small
transformer trained on in-context linear regression switches from memorizing its pretraining tasks
(dMMSE-like) to generalizing (ridge-like), under AdamW and Muon (SGD was tested and excluded; see below).
Results: [`results/e1-results.md`](results/e1-results.md).

## Reproduce the tables and figures (CPU, about a minute)
All logged runs (metrics, configs, reference values; no checkpoints) are in `results/runs/`.
```bash
pip install numpy matplotlib        # analysis needs only these
cp -r results/runs /tmp/runs && rm -rf /tmp/runs/analysis
RUNS_DIR=/tmp/runs python analyze.py                   # every M* table, r-per-M table, threshold
                                                       # robustness, race prediction, and the 3 plots
python jobs.py select --root results/runs --out /tmp/hparams.json   # full LR x WD sweep table
```
`/tmp/runs/analysis/summary.md` should match `results/runs/analysis/summary.md`, and the plots match
`results/plots/`.

## Setting
D=8, K=16 in-context pairs, label noise σ²=0.25, tasks w ~ N(0, I). Model: GPT-2 style,
4 layers, width 64, 2 heads, 9 numbers per token, 32 tokens, 202,752 parameters (`icl/model.py`).
Batch 256, 500K steps, linear warmup (5%) + cosine decay, grad clip 1.0.
Task pools are nested across M (pool for M = first M rows of one 2^15 draw per seed); seeds are
shared across optimizers (same pool, init, data stream, eval sets).

Weight decay is decoupled and written as a per-step shrink `omega` at peak LR, so it means the same
for every optimizer (`omega=1e-4` = AdamW lr 1e-3, weight_decay 0.1). Muon is applied to hidden
matrices (q, k, v, o, fc1, fc2); everything else uses Adam with `lr_aux=1e-3`.

Tuned values (`results/hparams.json`): AdamW lr 3e-3, wd 0; Muon lr 5e-3, wd 1e-4. No SGD setting
(lr 0.003-0.3, wd 0 or 1e-4) left the trivial "predict 0" solution in the 50K-step pilot, so SGD was
excluded by the proposal's rule (must come within 5% of ridge).

## Files
| file | what |
|---|---|
| `icl/model.py` | transformer, token layout, Muon/decay parameter masks |
| `icl/data.py` | task pools, sampling, ridge and dMMSE references |
| `icl/optim.py` | SGD / AdamW / Muon with shared schedule, clipping and weight decay |
| `train.py` | one resumable run; logs to `metrics.jsonl` and W&B (`cs182-proj/cs182-icl-optimizers`) |
| `launch.py` | packs runs onto GPUs, restarts crashed runs from checkpoint |
| `jobs.py` | job files for bench / sweep / E1; `select` picks tuned LR and WD |
| `analyze.py` | M\* per optimizer and seed, seed spread, clocks, race prediction, plots |
| `pipeline.sh` | sweep -> select -> E1 (M = 2^9..2^15) -> analysis, unattended |
| `extend_low.sh` | E1 at M = 2^3..2^8.5 with the same hyperparameters, then re-analysis |
| `tests/test_core.py` | parameter count, references vs numpy, causality, resume determinism |
| `results/` | logged runs, plots, summary, selected hyperparameters, pipeline log |

## Full rerun from scratch (8x A100 40GB, about 7 hours)
```bash
git clone https://github.com/nayan-ramakrishnan/cs182proj.git && cd cs182proj
bash setup.sh                     # venv, JAX+CUDA, CPU tests
wandb login                       # optional; runs continue without it
export RUNS_DIR=/path/to/persistent/storage/cs182runs
tmux new -d -s pipeline 'bash pipeline.sh'      # sweep (26 runs) + E1 high-M grid (36 runs) + analysis
# when it finishes:
tmux new -d -s low 'bash extend_low.sh'         # E1 low-M grid (28 runs) + analysis over all 64
watch -n 60 python status.py                    # progress
```
Rerunning any of these is safe: finished runs are skipped, unfinished ones resume from checkpoint.
GPU results reproduce up to floating-point nondeterminism; package versions are not pinned.
