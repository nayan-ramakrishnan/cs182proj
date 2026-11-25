# E1 results (run 2026-11-24, Lambda 8xA100, code at commit f4a7103)

Setup: D=8, K=16, sigma^2=0.25, 4L/width 64/2 heads (202,752 params), batch 256, 500K steps, warmup+cosine, clip 1.0.
LR and WD tuned per optimizer in a 50K-step sweep at M = n_train, then held fixed across every M and seed:

| optimizer | lr | weight decay (per-step, at peak lr) | sweep: new-task MSE over ridge |
|---|---|---|---|
| AdamW | 3e-3 | 0 | +0.93% (wd 1e-4 gave +0.94%, a tie) |
| Muon | 5e-3 (Adam aux lr 1e-3) | 1e-4 | +0.52% (wd 0 gave +0.71%) |

SGD dropped: none of its 10 sweep settings (lr 0.003-0.3 x wd 0/1e-4) learned anything in 50K steps. Every one ended at
new-task MSE 1.0242 (+172% over ridge), i.e. still predicting ~0, the trivial solution, so the proposal's rule removed it.
Full sweep table: `pipeline.log`.
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
- Not directly comparable to Raventos et al.'s threshold. Our model matches their "small PT" (4 layers, width 64, 2 heads; they report ~2^11.5 for it), but they define the threshold as the M where increasing batch size (256 -> 512) stops pushing the model toward dMMSE and starts pushing it toward ridge. We use the proposal's single-run readout (smallest M with Delta_ridge <= Delta_dMMSE). Training also differs (theirs: Adam, lr 1e-3, one-cycle schedule with 50% warmup; ours: per-optimizer tuned lr, 5% warmup + cosine). Their base model tracks dMMSE up to M of about 2^6, consistent in spirit with our ~2^7 crossover.
- Muon learns ICL ~2x faster (T_icl) yet memorizes at larger M; the simple race model underpredicts M* by about 3-4 log2 units and gets the ordering only marginally.
- Only 2 seeds.
- Confound: Muon ran with weight decay 1e-4, AdamW with 0. Raventos et al. found weight decay lowers M* (for Adam). If that holds for Muon too, decay pushed Muon's M* down, so the measured gap would be understated rather than created by decay; untested. Clean control: Muon with wd 0 (or AdamW with wd 1e-4) at M = 2^6 ... 2^9, 2 seeds.
- Plots: `plots/delta_vs_M.png`, `plots/memorization_curves.png`, `plots/clocks.png`; full numbers in `summary.json`.
