# 22 — PC-ALM feedforward predictor (credit-assignment test)

Host job 4608eecb (2026-10-16, commit e0b5e7f). Spec: wiki 'Method - PC-ALM predictor arm'.
Same network, data, optimiser and tests as the ff arms of findings/20; only the weight gradient differs.

Selection (seed 0, 90/10 train holdout): alpha in {0.1, 0.25, 1.0} x T in {20, 100} -> T = 100, alpha = 1.0
(held-out nmse 0.3042; next best 0.3275). alpha = 1.0 is the top of the grid (edge, flagged, not re-tuned).

Test nmse, k = 1 seen view, mean +- sd over 3 seeds:

| arm | clean | noise 0.1 | noise 0.2 | occlude 25% |
|---|---|---|---|---|
| backprop                 | 0.2992 +- 0.0006 | 0.6864 | 1.4694 | 1.8305 |
| plain PC, T = 20 (old)   | 0.3321 +- 0.0017 | 0.7282 | 1.6494 | 1.8501 |
| plain PC, T = 100        | 0.3416 +- 0.0008 | 0.7403 | 1.6928 | 1.8508 |
| **PC-ALM, T = 100**      | **0.3088 +- 0.0008** | 0.6935 | 1.4776 | 1.8384 |

- PC-ALM closes 77% of the gap to backprop against plain PC at the same T (71% against the old T = 20 arm).
- Clean: NOT parity (gap 0.0096, about 12x the seed sd). Under every corruption: within max(sd) of backprop, i.e.
  parity by the pre-registered rule.
- Credit: cosine to the BP gradient starts at 0.98 and drifts to 0.77-0.91 by step 6000 (norm ratio 0.48-0.94);
  plain PC sits at 0.29-0.70 with norm ratio 0.003. Pre-registered bar (> 0.95 throughout) is NOT met after step 1000.
- More inference hurts plain PC again (T = 20 -> 100: 0.332 -> 0.342).
