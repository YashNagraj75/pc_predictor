# 29 — Residual feedforward PC-ALM, depth 10, rerun with correct gradients

Replaces the INVALID findings/25 PC-ALM d10 arm. Host jobs dc7a9a37 (selection + A s0, s1) + cbd08078 (rest), commit
fa672a4/12b318f, NVIDIA_TF32_OVERRIDE=0 + XLA_FLAGS=--xla_gpu_autotune_level=0, audit gate cos 1.0000. No BP trained.

Selection on the 90/10 holdout:
- A (pre-registered, rho 1): alpha {0.25,1} x T {20,100} x lr {1e-3,3e-3,1e-2} -> T 100, alpha 1, lr 1e-3 (val 0.3596; lr at grid EDGE)
- R16 (new arm from the credit check): rho 16, alpha 1, T 400, lr {1e-3,3e-3,1e-2} -> lr 1e-2 (val 0.2964; lr at grid EDGE)

Test nmse, one view seen, mean (sd) over 3 seeds:

| arm | clean | noise 0.2 | occlude 25% | credit cos (end) | worst layer (end) |
|---|---|---|---|---|---|
| A, rho 1 | 0.3616 (0.0022) | 1.569 | 1.778 | 0.79 | 0.70 |
| **R16, rho 16, T 400** | **0.2979 (0.0003)** | **1.411** | 1.892 | 0.88 | 0.86 |
| old invalid E1 PC-ALM d10 (faulty gradients) | 0.388 | | | | |

Other-network references (NOT same-network controls): plain 3-layer MLP (findings/22) BP 0.2992, PC-ALM 0.3088,
BP noise 0.2 1.469, occlusion 1.831.
- The penalty matters: rho 16 beats rho 1 by 0.064, and credit holds (worst layer 0.86 vs 0.70).
- R16 is the lowest predictor error in the project, and lower than the plain-MLP BP by 0.0013 (about 2 sd of either).
  BUT the frozen residual BP d10 reference trained on faulty gradients (findings/27), so there is NO same-network
  BP comparison; the plain-MLP BP is a different, shallower network.
- lr 1e-2 is the top of the grid, so R16 may not be at its best.

## Same-network BP reference (2026-10-27)

Residual BP d10, standard training, lr 1e-3 fixed (user rule: no tuning), retrained with the fix (job ab3c17c4):
0.3511 / 0.3514 / 0.3513 (sd 0.0001), noise 0.2 1.543, occlusion 1.811. Identical to the original frozen reference
(0.3513), which therefore STANDS and stays the reference; the retrain is a verification, kept in bp_retrain_check/.

| vs BP (paired by seed) | clean | noise 0.2 | occlusion 25% |
|---|---|---|---|
| PC-ALM rho 16, T 400 (lr 1e-2) | **-0.053** (all 3 seeds): WIN | 1.411 vs 1.543: better | 1.892 vs 1.811: worse |
| PC-ALM rho 1, T 100 (prereg, lr 1e-3) | +0.008 / +0.011 / +0.012: LOSS | 1.569: worse | 1.778: better |

Caveat: PC-ALM rho 16 selected lr 1e-2 (grid top); BP is fixed at lr 1e-3. On this muPC network that lr asymmetry may
explain much of the win. Separating rule from step size needs a diagnostic BP run at lr 1e-2 (not a reference).
