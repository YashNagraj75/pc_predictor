# 21 — Generative PC predictor, stage 1 (seed 0)

Host job ec7a4426 (2026-10-13/14, commit 7468692). 15 runs, 0 diverged. Frozen backprop backbone bp_sigreg_etafix_s0.
Spec: wiki 'Method - Generative PC predictor plan' (revision 2); prereg gpc_prereg.json.
n = 1 seed per arm: every comparison here is provisional.

Test nmse, k = 1 seen view, T = 400 (lower is better):

| network        | backprop auto-decoder | PC     |
|----------------|-----------------------|--------|
| depth 3, zero  | 0.3059                | 0.3812 |
| depth 3, amort | 0.2984                | 0.3804 |
| depth 10, zero | 0.2780                | 0.3753 |
| depth 10, amort| 0.2826                | 0.3748 |
| d10 PC, hidden precision 9 | —         | 0.3777 |
| d3 width 256   | 0.3384                | 0.3789 |

References (findings/20, 3 seeds): backprop set predictor 0.2937, feedforward backprop 0.2992, feedforward PC 0.3321, copy 0.5018.

- PC loses to backprop on the identical network at every depth/start (gap 0.075-0.10).
- Depth helps backprop (0.306 -> 0.278, now the best predictor in the project) and barely moves PC (0.381 -> 0.375).
- Amortised start (Fujita) changes PC by < 0.001 at T = 400. Hidden precision 9 does not close the gap.
- PC's error is flat across depth, width (600 vs 256) and precision: a bottleneck that is not capacity.
- PC is best at 25-50 test inference steps and gets worse with more (0.363 -> 0.376 at T = 2000).
- Robustness: PC is worse than backprop in absolute error under every corruption; its smaller ratio to clean
  error comes from a worse clean error.
- Diagnostics: PC's belief s has norm 0.04 (d10) vs 0.12-2.2 for backprop. At depth 3, 95% of PC's settled energy sits at
  the top link (x_H vs the prediction from s), 0.8% at the observation: the hidden levels absorb the seen views and only a
  weak residual reaches s. s still carries information (shuffling s: 0.33 -> 1.73).
