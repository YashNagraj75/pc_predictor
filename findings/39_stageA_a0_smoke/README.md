# 39 — Stage A smoke test and A0 (cost/credit choice of T and power iterations)

> **INVALIDATED 2026-11-06 (findings/41):** trained and/or credit-measured with XLA GPU autotuning ON, which gives wrong
> batched gradients (rel error 0.46-0.61 vs CPU). Evaluation is correct, the training gradients were not. Rerun pending.

Host job 93ef0bdf (commit e4813d8), 2026-11-03. Smoke: every arm at depth 6 (6 steps, exact resume to 12) and ePC
depth 10 ran with rc 0; peak GPU memory 2.4 GiB (depth 6 ePC), 4.1 GiB (depth 10 ePC).

A0: depth-10 predictor trained 2 epochs by BP on the frozen V0 features; gain lambda_max(J^T J) median 1892 (3 draws of
64 images; per-draw medians 1646-2086, max up to 6261). Credit = cosine of the arm's weight gradient to BP's, mean over draws.

| arm | T 20, 12 its | T 20, 4 its | T 10, 12 its | T 10, 4 its | s/step (T20/12 -> T10/4) |
|---|---|---|---|---|---|
| ePC-ALM, rho = gain/4 | 0.976 (min leaf 0.65) | 0.975 (0.64) | 0.975 (0.66) | 0.975 (0.66) | 14.4 -> 6.9 |
| ePC, rho = gain/4 | 0.864 (0.23) | 0.863 (0.23) | 0.880 (0.35) | 0.878 (0.34) | 14.3 -> 6.9 |
| ePC, rho = gain/16 | 0.689 (0.05) | 0.686 (0.04) | 0.709 (0.23) | 0.708 (0.21) | 14.3 -> 7.0 |
| ePC, rho = gain/16, T 100 | 0.626 (-0.12) | | | | |

BP: 0.595 s/step. Pre-set rule (cos within 0.005 and weakest leaf within 0.01 of T20/12 for every arm) -> **T 10, 4
power iterations**: ePC cost halves (about 11.7x BP per step).
- At depth 10 the ALM arm at gain/4 is close to, but not exactly, BP: cos 0.975 with the weakest leaf at 0.65 (at depth
  6, after 150 BP steps on one batch, the gate gave 1.0000). So "ALM == BP" no longer holds by construction here.
- For the no-multiplier arms credit depends on T: longer relaxation moves further from BP (k16: 0.71 at T 10, 0.69 at
  T 20, 0.63 at T 100). T 10 therefore gives their mildest departure from BP; this is part of the arm definition.
