# 49 — Latent analysis with three seeds of gain-matched ePC, depth 10 (2026-11-26)

A4000 tmux job latent_p3 (commit def8037, 3 reviewers no blocker), GPU gate d10/d6 pass, tests pass, output check PASS.
All 24 models re-analysed at one commit; every trained model reproduces its report. New: epc_alm_blk4 d10 seeds 1-2.
Ranges are over seeds (BP 5, ePC 3); standard PC and the other ePC arms are one seed. Raw JSON in raw/.

| depth 10 | BP (5 seeds) | gain-matched ePC (3 seeds) | holds on every ePC seed? |
|---|---|---|---|
| noise, clean-fitted scale | 1.116–1.129 | 0.921–1.030 | yes |
| noise, direction only | 0.839–0.844 | 0.812–0.833 | yes (small) |
| movement along real noise | 0.787–0.791 | 0.692–0.745 | yes |
| worst-case Jacobian | 1.607–1.633 | 2.430–3.389 | yes (larger) |
| cost of removing nearby context, minus random | 0.275–0.308 | 0.208–0.232 | yes (PC relies less on near context) |
| attended distance / uniform | 0.655–0.670 | 0.697–0.745 | yes (less local) |
| error / variance, top band | 0.107–0.107 | 0.111–0.119 | yes |
| error / variance, finest band | 0.526–0.527 | 0.585–0.832 | yes (gap grows with the seed's clean error) |

CKA to BP seeds 1-4 (position-residualised, q1_latent_cka_seeds_d10.csv):
- Middle and late layers diverge on every seed: nodes 13-17 ePC 0.65-0.77 (BP seed to seed 0.98).
- Re-convergence at the output is NOT robust: last node 0.93 / 0.92 / 0.39 (seed 2 does not re-converge). Drop that claim.
- ePC seeds agree with each other much more than with BP in the middle layers (0.86-0.98 at nodes 1-15): PC reaches a
  consistent solution of its own, not random divergence.

Standard PC (one seed): its hidden states are the untrained network's (CKA to init 0.97-1.00 at nodes 1-19, 0.66 at the
last node). Only the output layer learned, which is credit starvation seen directly. This explains its kNN oddity: kNN of its
predictions 0.562 equals the context-embedding floor 0.565, because the context passes almost untouched through the residual
stream; trained predictors (BP 0.548, ePC 0.539-0.550) sit slightly below that floor. It is not a richer representation.

Paper consequences: the noise, sensitivity, context-reliance, attention and fine-direction results hold on 3 seeds; "re-converges
near the output" goes; "ePC seeds share a PC-specific solution" is new; the d6-vs-d10 depth trends stay single-seed (not claimed).
