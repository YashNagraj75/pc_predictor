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

## Follow-up diagnostic (diag_c3, host job 7028d816, commit 9e20e0e; inference only) — `diag_c3/`

Belief decomposition, test k=1 nmse on each trained decoder (mean over seeds): own rule T=400 / own rule T=2000 / exact
auto-decoder belief T=2000:
- d3 BP 0.305 / 0.308 / 0.308; PC-ALM rho4 0.352 / 0.401 / 0.373; no-mult 0.382 / 0.389 / 0.479; plain PC 0.381 / 0.387 / 0.482.
- d10 BP 0.279 / 0.303 / 0.357; plain PC 0.377 / 0.377 / 0.478.
=> A more converged belief is WORSE for every decoder: each decoder is tuned to its own truncated (T=400) inference.
   The "belief is the bottleneck" hypothesis is not supported; s_dist to a settled AD belief was a misleading yardstick.

Depth-10 loop grid (alpha {1,.5,.25,.1} x k_inner {1,4,16} x c {1,.5,.25}, PC-ALM rho=4, T=400, 512 test samples, k=1)
on trained depth-10 decoders (PC-ALM s0 at steps 1000/2000/final, no-mult s0, BP s1) and a d3 reference:
- best credit cos on ANY depth-10 decoder, ANY schedule: 0.64 (vs 0.79 for the trained d3 decoder; 0.965 at init in diag_c2).
- smaller dual rates / more inner steps reduce oscillation only by not converging within T=400 (held-out nmse 10-50).
=> No multiplier schedule in the grid fixes depth 10; the deep trained decoder's inference problem is the obstacle.

## Qi et al. fixes, inference-only check (diag_c4, host job 93f1dca1, commit 7129350) — `diag_c4/`

Trained decoders (d10: plain PC s1/s2, BP s1, no-mult s0, PC-ALM s0 step 1000; d3: PC s1, BP s1, PC-ALM s0), 512 test
samples, k=1. Gates passed (schedule 'none' == pc_infer; rule 'pc' == pc_wgrads). d10 plain PC, mean of 2 seeds:

| T | spike | held-out nmse | cos to BP-AD grad: standard / forward-update qi / fixed | energy imbalance log10(max/min) |
|---|---|---|---|---|
| 400 | none | 0.360 | 0.233 / 0.067 / 0.096 | 1.6 |
| 400 | step 1 at wavefront | 719 | 0.643 / 0.273 / 0.298 | 2.4 |
| 11 | none | 0.986 | 0.044 / 0.049 / 0.084 | 3.3 |
| 11 | step 1 at wavefront | 3.7e6 | -0.153 / 0.175 / 0.964 | 4.6 |

- Spiking precision destroys the belief (held-out nmse 10^2-10^13 at every budget). Our curvature-normalised activity
  step (c=2: 1.8/(1+2L)) is already near the stability limit, so a spike has no headroom; Qi's spike compensates for
  their small step (alpha ~ 0.05). The near-exact credit at T=H+1 (cos 0.96, the one-sweep backprop effect) is
  measured at a garbage belief.
- The forward-update rule gives LOWER agreement than the standard rule on PC-trained decoders (d10 0.10 vs 0.23;
  d3 0.22 vs 0.67); it helps only on the ALM-trained d3 decoder (0.72 vs 0.26).
- The energy imbalance is mild (about 1.6 orders of magnitude at d10, T=400), not the 10^18 Qi report for
  discriminative VGG10, so their diagnosis does not describe our failure.
=> Pre-registered decision rule (improve d10 agreement without worsening held-out error) is met by neither fix: no
   training run. Caveat: inference on decoders not trained with the fixes; weights co-adapting with spikes is untested.
