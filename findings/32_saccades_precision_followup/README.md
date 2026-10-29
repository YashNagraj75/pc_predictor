# 32 — Saccade follow-up: class probe, precision pooling, error-reduction planner, BP at equal view counts

Host job 8ad1f8f8 (commit cc11bd8, rev 3 trimmed), 2026-10-28/29, inference only. 63 + 55 gates passed. 2000 test images;
PC predictors 3 seeds; frozen BP auto-decoder 2 seeds (seed 0 weights never saved); 'Gmap' and the T-vs-4T check seed 1 only.
BP rows: augmented whole views 1-192, unchanged weights, no glimpses (user decision 2026-10-28). nmse lower is better;
kNN = cosine k=20 class accuracy of the predicted target latents (true targets 0.376, mean latent 0.118).

- PC vs frozen BP at EQUAL view count (augmented whole views): clean, generative BP wins (d10 192 views 0.169 vs 0.193),
  feedforward PC wins (0.227 vs 0.264, BP at fixed lr 1e-3). Occlusion, generative PC wins at >= 6 views (d10 1.533 vs
  1.623, d3 1.563 vs 1.630); feedforward BP slightly ahead (1.352 vs 1.368). Noise: mixed, within ~0.03.
- Class accuracy (kNN) is the same for every model and pooling within a condition (clean ~0.39, noise ~0.26, occlusion
  ~0.17): no robustness change costs class information, and under occlusion no model keeps much class information.
- Precision pooling (weight 1/(leave-out error + 0.1 median)): noise -0.02 to -0.04, occlusion -0.04 to -0.08, clean ~0.
  Shuffled weights give nothing. Leave-one-glimpse-out == leave-one-crop-out. It beats the occlusion oracle (drop
  occluded crops), so it down-weights more than the occluder.
- Planners: ig == ig_shuf == epi == epi_shuf == ior == bu within 0.002 everywhere; same fixations (occluded fraction 0.387 at
  K=32 for all; random 0.401, oracle 0.323). Only K=1 under occlusion helps (planner first glimpse at the bright centre,
  1.21 vs random 1.67, == oracle). Under noise the planners are worse than random at K=32.
- Saccades (32 glimpses, ~192 latents, best pool) still lose to 192 whole views with precision pooling in every condition.
- PC-trained generative weights under BP's own inference (Gmap, seed 1): under occlusion the PC amortised belief 1.552 <
  Gmap 1.596 < BP 1.630 (d10, 192 views): both the amortised inference and the PC-trained weights contribute. Clean: BP
  weights better (0.169 vs 0.190).
- BP inference convergence: 4x the steps changes BP's nmse by <= 0.04 and not in BP's favour under corruption.
