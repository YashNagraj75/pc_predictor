# 33 — MNIST: the two best PC predictors vs frozen BP (Galaxy10 settings, matched 15,675 training images)

Host job 8e68f445 (commit 3a61799), 2026-10-30/31. 39/39 gates; 12/12 runs rc 0; eval rc 0. 3 seeds for every model
(BPg seed 0 weights saved this time). Frozen MNIST encoder (BP LeJEPA+SIGReg, 17,200 steps, latents only): kNN 0.9175 vs
0.898 untrained (class-error cut 19%; Galaxy's own encoder 3.8%); raw pixels 0.947. Test nmse, lower is better.

| predictor | input | clean PC / BP | noise PC / BP | occlusion PC / BP |
|---|---|---|---|---|
| Generative amortised PC-ALM d10 | 1 view | 0.206 / **0.183** | 0.331 / **0.312** | **1.123** / 1.142 |
| same | 192 views, precision pool | 0.118 / **0.094** | 0.157 / **0.137** | **1.096** / 1.114 |
| Residual ff PC-ALM d10 (lr 1e-2 vs BP 1e-3) | 1 view | **0.191** / 0.246 | **0.306** / 0.355 | 1.123 / **1.116** |
| same | 192 views, precision pool | **0.125** / 0.167 | **0.167** / 0.214 | 1.077 / **1.050** |

Paired by seed (PC - BP): G clean +0.021..+0.026, noise +0.015..+0.022, occlusion -0.013..-0.022 (1 view) and
-0.014..-0.021 (192 views); F clean -0.055 / -0.042, noise -0.049 / -0.048, occlusion +0.005..+0.008 (1 view, under 0.01:
no detectable effect) and +0.025..+0.031 (192 views). Seed sd <= 0.006 everywhere.

- Same pattern as Galaxy for clean and noise: F beats BP (lr caveat as findings/29), G loses to BP.
- Occlusion flips by model: G beats BP at every view count (on Galaxy only with multiple views); F loses to BP at 192 views
  (on Galaxy F won). Precision pooling hurts F under occlusion on MNIST (1.077 vs mean pool 1.063); on Galaxy it helped.
- Caveat: MNIST occlusion errors are all ~1.1 and kNN ~0.23 for every model: the black occluder equals MNIST's background,
  so it deletes ink without an edge, and predictions are close to uninformative. Treat occlusion differences on MNIST as
  differences between near-failed predictors (review DS-6).
- kNN class accuracy of the predictions is the same for PC and BP in every condition (clean ~0.92, noise ~0.91).
- BP inference settled: 4x steps changes BPg nmse by <= 0.021.
