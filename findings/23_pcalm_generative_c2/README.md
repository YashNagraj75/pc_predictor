# 23 — PC-ALM generative predictor (C2), depths 3 and 10

Host jobs c8634b25 (commit a8402d3, rev-2 design) and 22c2dca3 (post-hoc c = 0.5 amendment, user-approved), 2026-10-17/18.
Spec: wiki 'Method - PC-ALM generative predictor (C2)'; prereg gpc_alm_prereg.json (+ gpc_alm_amend_c05_prereg.json in Q1-PhaseA).
Seed 0 of BP and plain PC is stage 1 (findings/21), reused after det_check passed for all four.

In-job choice: rho_hi = 4, T2 = 100 at both depths. c: d3 rho=1 0.25, rho=4 0.5; d10 1.0 for both (grid top).

Test nmse, k = 1 seen view, T = 400, mean (sd) over seeds:

| depth | backprop | plain PC | rho=4, no multipliers | PC-ALM rho=4 | PC-ALM rho=1 |
|---|---|---|---|---|---|
| 3  | 0.3053 (0.0008) | 0.3812 (0.0004) | 0.3818 (0.0013) | **0.3522 (0.0053)** | 0.4666 (0.0523) |
| 10 | 0.2783 (0.0008) | 0.3763 (0.0010) | 0.3828 (0.0007) | 0.98 (2 runs) + 1 failed | 1.01 (2 runs) + 1 failed |

Depth 10 amendment, c = 0.5: seeds 0 and 1 failed at c = 0.5, 0.25 and 0.125; seed 2 survived only at c = 0.125, nmse 0.638.

- Depth 3: paired by seed, PC-ALM rho=4 beats the same network without multipliers by 0.0296 (0.0292/0.0258/0.0339)
  and closes 31-45% of plain PC's gap to BP. Not parity: 0.0468 behind BP (0.0463/0.0531/0.0411).
- Precision alone does nothing (no-multiplier rho=4 vs plain PC +0.0006 at d3).
- Credit (cos to the BP auto-decoder gradient, fixed probe): rho=4 about 0.98 to step 2000, then 0.81-0.89 at the end;
  rho=1 falls to about 0.3-0.6 within 1000 steps.
- More inference hurts PC-ALM: settled T=2000 0.401 vs 0.352 at T=400 (d3).
- No robustness advantage: noise 0.2 2.21 vs BP 2.01; occlusion 2.26 vs 2.20 (d3).
- Depth 10: the multiplier loop is unstable at every step size tried (c = 1 down to 0.125); without multipliers the same
  network is stable at c = 1 (0.383). BP improves with depth (0.305 -> 0.278); no PC variant does.
