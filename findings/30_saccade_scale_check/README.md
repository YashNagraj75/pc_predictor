# 30 — Saccade plan S0: how small can a glimpse crop be? (2026-10-27)

Host job e5dc2719 (commit 5cb3fdc), `Q1-PhaseA/diag_scale.py`, GPU fix on. Inference only; no BP model touched.
Backbone `runs/bp_sigreg_etafix_s0` (re-encodes the stored test latents, rel err 6.9e-4).
One crop per test image (2000) at fixed area, random position, no rotation/flip/colour; the PC predictors predict all
4 stored target views from that single crop. nmse mean over 3 seeds (sd <= 0.006 for area >= 0.3).

| crop area | linear scale | OOD (frac above in-dist p99 Mahalanobis) | kNN | G d3 | G d10 | F d10 |
|---|---|---|---|---|---|---|
| 0.05 | 0.22 | 0.10 | 0.156 | 1.603 | 1.678 | 1.624 |
| 0.10 | 0.32 | 0.13 | 0.172 | 1.637 | 1.679 | 1.586 |
| 0.20 | 0.45 | 0.20 | 0.204 | 1.533 | 1.509 | 1.340 |
| 0.30 | 0.55 | 0.10 | 0.265 | 0.902 | 0.848 | 0.727 |
| 0.40 | 0.63 | 0.03 | 0.328 | 0.493 | 0.442 | 0.400 |
| 0.60 | 0.77 | 0.006 | 0.374 | 0.320 | 0.294 | 0.282 |
| 0.80 | 0.89 | 0.004 | 0.393 | 0.276 | 0.257 | 0.256 |
| 1.00 | 1.00 | 0.003 | 0.384 | 0.274 | 0.255 | 0.257 |

Reference: stored-view kNN 0.3725; standard k=1 test (one augmented view -> other 3) G d3 0.331, G d10 0.3075.

- Below area 0.3 the predictions are worse than predicting the mean (nmse > 1): those crops are unusable.
- Area 0.4 (the bottom of the backbone's training range) is the smallest usable scale, but one such crop already
  costs a lot (G d10 0.44 vs 0.29 at area 0.6).
- Decision D1: foveal crops at area 0.4 (scale 0.63, fixations within +-0.37); parafoveal 0.6; peripheral 1.0.
  Area 0.3 is kept only as an exploratory variant.
