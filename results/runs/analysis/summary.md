# E1 summary

## M* (end of training)

| optimizer/seed | runs | seen: grid | seen: interp | new: grid | new: interp |
|---|---|---|---|---|---|
| adamw/s0 | 16 | 2^7.00 | 2^6.82 | 2^5.00 | 2^4.70 |
| adamw/s1 | 16 | 2^7.00 | 2^6.95 | 2^7.00 | 2^6.04 |
| muon/s0 | 16 | 2^8.00 | 2^7.84 | 2^8.00 | 2^7.17 |
| muon/s1 | 16 | 2^8.00 | 2^7.80 | 2^8.00 | 2^7.09 |

## End-of-training r = ΔdMMSE/(ΔdMMSE+Δridge) on seen tasks (mean over seeds; >0.5 = ridge-like)

| optimizer | 2^3 | 2^4 | 2^5 | 2^6 | 2^7 | 2^8 | 2^8.49985 | 2^9 | 2^10 | 2^10.4998 | 2^11 | 2^11.4998 | 2^12 | 2^12.5001 | 2^13 | 2^15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adamw | 0.00 | 0.00 | 0.00 | 0.08 | 0.58 | 0.73 | 0.80 | 0.86 | 0.93 | 0.94 | 0.95 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| muon | 0.00 | 0.00 | 0.00 | 0.02 | 0.13 | 0.60 | 0.67 | 0.73 | 0.89 | 0.92 | 0.95 | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 |

## Crossover at other thresholds (seen tasks; log2 M where r first exceeds the threshold, per seed)

| threshold | adamw | muon |
|---|---|---|
| r = 0.25 | 6.28, 6.40 | 7.27, 7.23 |
| r = 0.5 | 6.78, 6.92 | 7.80, 7.76 |
| r = 0.75 | 8.20, 8.08 | 9.16, 9.08 |

## Optimizer differences vs seed spread (seen tasks, log2 M)

| pair | delta | seed spread | exceeds? |
|---|---|---|---|
| muon - adamw | +0.93 | 0.13 | yes |

## Race prediction (eps_icl=0.1, r_mem=0.25)

| optimizer | T_icl | memorizing runs | slope | predicted M* | measured M* (mean) |
|---|---|---|---|---|---|
| adamw | 8950.0 | 8 | 1.46 | 2^3.56 | 2^6.89 |
| muon | 4300.0 | 10 | 1.64 | 2^3.67 | 2^7.82 |
