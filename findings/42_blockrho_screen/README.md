# 42 — Per-block rho: gates and inference-only credit screen (correct numerics)

Host job 614ff99b (commit 504e742, Triton GEMMs off; GPU-vs-CPU gate PASS at d10 and d6, all rel <= 2.4e-6).
Checkpoints: the depth-10 ePC-ALM and BP predictors of findings/40 (trained under the findings/41 fault, so unusual
weights; the credit measured here is correct). 5 draws x 64 images x 4 blocks, T 10, ALM alpha 1. Table: q1_blockrho_screen.csv.

Gates: per-block path == commit 421f84f bitwise for the scalar arms; per-block rho == max(1, own gain/4); batched ==
sum of single blocks at their own rho (rel 1e-6, rho spread 96x); BP limit (rel err 3e-5 at 1e4 x gain). One gate is
ill-posed: it assumed blocks with gain < 4 at init (none at depth 10; init gain ~20). 4-iteration gain estimate: 5th
percentile 0.76-0.85 of the 100-iteration value, minimum 0.44; no block in the predicted unstable range for k in {1,2,4}.

| rule (ALM ckpt / BP ckpt) | cos to BP | size ratio | weakest leaf | median rho |
|---|---|---|---|---|
| median gain / 4 (the planned rule) | 0.915 / 0.997 | 0.73 / 1.04 | 0.83 / 0.99 | 674 / 105 |
| max gain / 4 (one scalar rho) | 0.9996 / 0.9997 | 0.97 / 1.00 | 0.998 / 0.997 | 8578 / 938 |
| per-block gain / 4 | 0.9985 / 0.9987 | 1.04 / 1.06 | 0.99 / 0.99 | 543 / 90 |
| per-block gain / 2 | 0.9985 / 0.9978 | 0.94 / 0.96 | 0.97 / 0.98 | 1087 / 181 |
| per-block gain / 1 | 0.9998 / 0.9997 | 0.97 / 0.99 | 0.999 / 0.994 | 2174 / 362 |
(12 power iterations instead of 4 change nothing.)

- With correct gradients the planned median rule is already at cos 0.915 on the hardest checkpoint, not 0.44 (that
  number in findings/40 was the fault). Its weakness is a gradient 27% too small on the high-gain blocks.
- Both fixes bring credit to ~0.999 with the right size: per-block rho, and simply a larger scalar rho (max rule).
  Per-block rho gets there with a smaller median rho (543 vs 8578).
- Pre-set decision rule did NOT pass for any configuration, for two reasons: (1) its energy criterion ("energy lower at
  the end in >= 99% of blocks") is ill-posed for ALM: with multipliers the augmented energy rises by construction, and
  it "fails" (~100%) for every rule including the median one; this was my design error. (2) the gain over the median
  rule was >= 0.05 cos on 4 of 5 draws; the smallest was 0.037. The automatic decision was not written (pandas import
  crash after the screen); no training was started.
