# 47 — Prior-art check for the four claims from the latent analysis (2026-11-13)

Four sub-agents, one per claim; ~160 papers screened (web search, arXiv API, OpenAlex for some), 20+ read in full. Every
arXiv id below was re-checked against the arXiv API (all 14 load-bearing ids exist with the stated titles).
Table: lit_survey_latent_claims.csv (45 rows: partial / related; none rated "anticipates").

| claim | verdict | closest prior work | what stays ours |
|---|---|---|---|
| 1 gain-relative per-block precision trains a PC transformer | partly anticipated | Qi et al. 2506.23800; muPC 2505.13124; ePC 2505.20137; Error Highways 2606.22744; Baskakovs et al. 2605.20293 (identity precision causes the depth failure; learned per-layer precision fixes it) | the diagnosis is NOT new. New: precision set from the measured output gain lambda_max(J^T J), per block, re-measured in training; first in a transformer predictor |
| 2 noise: less movement along real noise despite a larger worst-case Jacobian | partly anticipated | Predify 2106.02749; Alamia et al. 2106.04225; CNN-F 2007.09200 (PC feedback at test time on BP backbones is more noise-robust); Oliviers et al. 2605.11911 (directional "target alignment", training time) | PC as the learning rule at matched lr (not a test-time add-on); the worst-case vs real-noise-direction split; the scale-refit control |
| 3 different internal solution, same output | partly anticipated (incremental) | Li et al. 2304.13098 (SNN vs ANN CKA: diverge mid, re-converge); Lance et al. 2605.08564 (feedback alignment converges to BP); Li et al. 2609.00686 (PC-BP gradient alignment < 0.7 in MLPs) | PC vs BP layer CKA in a transformer, with near-identical gradients; attention locality tied to the learning rule |
| 4 error deficit in fine directions, shrinking with depth | apparently new | spectral bias (Rahaman 1806.08734; Xu 1901.06523; deep F-principle 2007.14313); Saxe 1312.6120; PC curvature/trust-region theory (Innocenti 2408.11979, 2510.23323); Oliviers 2605.11911 (depth-dependent PC-BP gap) | the PC-vs-BP decomposition by target principal component and its depth scaling; frame as PC's instance of spectral bias |

Caveats: the precision sub-agent did not use OpenAlex and hit arXiv rate limits (covered by web search instead).
"Apparently new" means not found in this search, not proven absent.
