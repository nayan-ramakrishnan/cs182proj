# How does the optimizer shape generalization in in-context regression?

CS182 project (Kaul, Liu, Ramakrishnan, Calderon). Measures the task-diversity threshold M\*
(Raventós et al. 2023) under SGD (momentum), AdamW and Muon. Currently implements **E1**.

## Setting
D=8, K=16 in-context pairs, label noise σ²=0.25, tasks w ~ N(0, I). Model: GPT-2 style,
4 layers, width 64, 2 heads, 9 numbers per token, 32 tokens, 202,752 parameters (`icl/model.py`).
Batch 256, 500K steps, linear warmup (5%) + cosine decay, grad clip 1.0.
Task pools are nested across M (pool for M = first M rows of one 2^15 draw per seed); seeds are
shared across optimizers (same pool, init, data stream, eval sets).

Weight decay is decoupled and written as a per-step shrink `omega` at peak LR, so it means the same
for every optimizer (`omega=1e-4` = AdamW lr 1e-3, weight_decay 0.1). Muon is applied to hidden
matrices (q, k, v, o, fc1, fc2); everything else uses Adam with `lr_aux=1e-3`.

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
| `tests/test_core.py` | parameter count, references vs numpy, causality, resume determinism |

## Running on Lambda
```bash
git clone git@github.com:nayan-ramakrishnan/cs182proj.git && cd cs182proj
bash setup.sh                     # venv, JAX+CUDA, tests
wandb login
echo 'export RUNS_DIR=<persistent fs>/cs182runs' >> ~/.bashrc && source ~/.bashrc
mkdir -p jobs

# 1. speed check
python jobs.py bench --copies 6 --tag p6 > jobs/bench_p6.jsonl
python launch.py jobs/bench_p6.jsonl --gpus 1 --per_gpu 18 --wandb 0

# 2. LR x WD sweep (26 runs, 50K steps, M = n_train), then pick the best per optimizer
tmux new -s sweep 'python launch.py jobs/sweep.jsonl --per_gpu 4'   # after: python jobs.py sweep > jobs/sweep.jsonl
python jobs.py select             # writes hparams.json (fallback: hparams_fallback.json)

# 3. E1: 9 M values x 2 seeds x 3 optimizers
python jobs.py e1 --hparams hparams.json > jobs/e1.jsonl
tmux new -s e1 'python launch.py jobs/e1.jsonl --per_gpu 7'

# 4. analysis -> $RUNS_DIR/analysis/
python analyze.py
```
Rerunning any `launch.py` command is safe: finished runs are skipped, others resume.
Progress: `cat $RUNS_DIR/e1_status.json`.
