# 24 — Qi et al. (2025) fixes trained into the generative PC predictor (screen, seed 0)

Host job 010439e8 (commit e308f7a), 2026-10-18/19; 108 gates passed; 16 arms finished, 2 failed (lit1).
Prereg: qi_prereg.json. BP auto-decoder controls from findings/23 (3 seeds): d3 0.3053, d10 0.2783.
Test nmse, one view seen, T=400 (Qi-regime arms: T=H+1). nmse ~0.99 = predicting the mean (collapsed decoder).

| arm | d3 | d10 |
|---|---|---|
| plain PC, rerun in this job (c=2) | 0.3812 | 0.3752 |
| spike, step 1 at the wavefront (lit1, c=2) | failed (exploded; last finite 0.79) | failed (exploded; last finite 1.57) |
| small step c=0.1 (control) | 0.3959 | 0.6634 |
| spike, stable (c=0.1, c_spike=2) | 0.3961 | 0.5661 |
| forward update, qi (c=2) | 0.5110 | 0.9865 |
| forward update, fixed (c=2) | 0.4909 | 0.9912 |
| forward update, fixed, c=0.1 | 0.3743 | 0.9894 |
| stable spike + fixed FU (c=0.1) | 0.3735 | 0.9894 |
| Qi regime: T=T2=H+1, stable spike + fixed FU | 0.9514 | 0.9889 |

Readings (prereg classes: <0.01 none, 0.01-0.03 candidate, >0.03 reportable):
- Plain-PC rerun reproduces stage 1 (0.381 / 0.375), so stage-1 controls are consistent.
- lit1 spike: reportable failure at both depths (pred error 56 -> 25525 by step 1000).
- Stable spike vs its c=0.1 control: d3 +0.0002 (none; spike barely changes s, rel 0.03); d10 -0.097 (reportable
  gain over the control), but both are far worse than plain PC (0.566 vs 0.375): the small base step is the cost.
- Forward update at c=2: reportable harm at both depths; at d10 the decoder collapses to the mean.
- Forward update at c=0.1, d3: 0.3743 vs its control 0.3959 (-0.022, candidate) and vs plain 0.3812 (-0.007, none).
  The best d3 arm (spike + FU, 0.3735) closes 10% of the PC-BP gap. No arm reaches BP.
- Depth 10: every forward-update arm collapses (0.99). Their credit cosine to the BP auto-decoder ends at 0.94-0.95,
  but on a collapsed decoder, so high alignment here is not useful credit.
- Qi's own short-inference regime collapses at both depths.
- No arm improves from d3 to d10; BP does (0.305 -> 0.278).
Caveat: one seed, one configuration per fix (Qi tuned 200 trials per setting).
