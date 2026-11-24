# E1 results (run 2026-11-24, Lambda 8xA100, code at commit f4a7103)

Setup: D=8, K=16, sigma^2=0.25, 4L/width 64/2 heads (202,752 params), batch 256, 500K steps, warmup+cosine, clip 1.0.
LR and WD tuned per optimizer in a 50K-step sweep at M = n_train (values in each run's final.json config on the Lambda filesystem `cs182proj`, us-west-1, and in each W&B run's config).
SGD dropped: best sweep setting missed ridge-level loss (within 5%), per the proposal's rule.
Grid: M = 2^3 ... 2^15 (16 values), 2 seeds, AdamW and Muon. 64 runs, 0 failures.

## M* (smallest M where Delta_ridge <= Delta_dMMSE; interpolated in log2 M)
| optimizer/seed | seen: grid | seen: interp | new: grid | new: interp |
|---|---|---|---|---|
| adamw/s0 | 2^7 | 2^6.82 | 2^5 | 2^4.70 |
| adamw/s1 | 2^7 | 2^6.95 | 2^7 | 2^6.04 |
| muon/s0 | 2^8 | 2^7.84 | 2^8 | 2^7.17 |
| muon/s1 | 2^8 | 2^7.80 | 2^8 | 2^7.09 |

Muon - AdamW (seen tasks): +0.93 log2 M (Muon's M* is about 1.9x larger); seed spread 0.13, so the shift exceeds seed noise.

## End-of-training r = Delta_dMMSE / (Delta_dMMSE + Delta_ridge), seen tasks (>0.5 = ridge-like)
| opt | 2^3 | 2^4 | 2^5 | 2^6 | 2^7 | 2^8 | 2^8.5 | 2^9 | 2^10 | 2^10.5 | 2^11 | 2^11.5 | 2^12 | 2^12.5 | 2^13 | 2^15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adamw | 0.00 | 0.00 | 0.00 | 0.08 | 0.58 | 0.73 | 0.80 | 0.86 | 0.93 | 0.94 | 0.95 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| muon | 0.00 | 0.00 | 0.00 | 0.02 | 0.13 | 0.60 | 0.67 | 0.73 | 0.89 | 0.92 | 0.95 | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 |

## Race prediction (eps_icl=0.1, r_mem=0.25)
| opt | T_icl (steps) | memorizing runs | t_mem slope vs M | predicted M* | measured M* |
|---|---|---|---|---|---|
| adamw | 8950 | 8 | 1.46 | 2^3.56 | 2^6.89 |
| muon | 4300 | 10 | 1.64 | 2^3.67 | 2^7.82 |

## Takeaways / caveats
- Ordering M*_AdamW < M*_Muon matches H1 (the SGD part of H1 is untested).
- M* is about 2^7-2^8, far below Raventos's ~2^11.5; likely because this model is smaller (less capacity to memorize). Untested.
- Muon learns ICL ~2x faster (T_icl) yet memorizes at larger M; the simple race model underpredicts M* by about 3-4 log2 units and gets the ordering only marginally.
- Only 2 seeds; the tuned weight decay may differ between optimizers (weight decay is known to shift M*), so check the run configs before attributing the shift purely to the optimizer.
- Plots (delta_vs_M.png, memorization_curves.png, clocks.png) and summary.json are in /lambda/nfs/cs182proj/cs182runs/analysis/ on the persistent filesystem.
