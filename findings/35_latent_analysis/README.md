# 35 — Latent analysis: why generative PC beats frozen BP under occlusion (inference only)

Host job 206f0389 (commit da026e0), 2026-11-01. Gates 47/66/66 pass (reproduction of stored evaluations for every seed,
known-answer tests, settling of BP inference). 2000 test images; Galaxy10 seeds 1-2 (no seed-0 BP weights), MNIST and
CIFAR-10 seeds 0-2; K = 48 whole views (the stored G-BPg occlusion gap is complete by K = 6). Paired clean/corrupted views.
nmse_calibrated = 1 - <P,z>^2/(|P|^2|z|^2): error after the best single rescaling of the predictions (direction only).

## Main result: the occlusion advantage is prediction SIZE, not information

| occlusion, K=48 | Galaxy10 nmse / calibrated / alpha* | MNIST | CIFAR-10 |
|---|---|---|---|
| G (PC weights, amortised mean) | 1.527 / 0.979 / 0.16 | 1.099 / 0.951 / 0.37 | 0.903 / 0.766 / 0.57 |
| PC weights, mean of per-view MAP | 1.373 / 0.980 / 0.19 | 1.032 / 0.951 / 0.44 | 0.896 / 0.765 / 0.57 |
| PC weights, joint MAP | 1.423 / 0.980 / 0.18 | 1.034 / 0.948 / 0.44 | 0.938 / 0.771 / 0.54 |
| BP weights, mean of per-view MAP | 1.553 / 0.983 / 0.15 | 1.090 / 0.946 / 0.38 | 0.907 / 0.771 / 0.57 |
| BP weights, joint MAP (frozen BPg) | 1.623 / 0.983 / 0.14 | 1.115 / 0.946 / 0.36 | 0.948 / 0.768 / 0.53 |

- After the best rescaling, all five weight x pooling combinations are equal under occlusion (within 0.006 on every
  dataset). Per seed, G - BPg calibrated: Galaxy -0.003/-0.004, MNIST +0.004..+0.005 (BP slightly better), CIFAR
  -0.002..-0.003, while the raw gaps are -0.08..-0.11, -0.011..-0.020, -0.043..-0.047.
- Every model over-predicts under occlusion (alpha* 0.14-0.57 << 1): the corrupted views leave little target information
  (calibrated nmse 0.98 Galaxy, 0.95 MNIST, 0.77 CIFAR; CKA with targets 0.15 / 0.11 / 0.44, equal across models).
  The model whose predictions are SMALLER (closer to the mean) scores better. G's win is that its predictions are smaller.
- Both factors set the size. BP's joint MAP is the most expansive pooling: averaging per-view beliefs with the same BP
  weights recovers 0.066-0.074 (Galaxy), 0.021-0.030 (MNIST), 0.036-0.047 (CIFAR) of nmse per seed, i.e. most of the
  G-BPg gap on CIFAR and MNIST. PC-trained weights give smaller predictions on Galaxy and MNIST under any pooling; the
  amortiser partly undoes that (G 1.527 vs PC weights + per-view MAP 1.373 on Galaxy).
- No model down-weights the views the occluder changed most: rank correlation of a view's effective weight with its
  latent displacement is -0.10..+0.27 for every model, with no PC/BP pattern.
- Clean and noise: the generative gap to BP survives calibration (clean CIFAR 0.141 vs 0.106; MNIST 0.118 vs 0.096;
  noise CIFAR 0.203 vs 0.174), so BP's generative predictor carries more target information there. The feedforward PC
  win on clean/noise also survives calibration (clean CIFAR 0.144 vs 0.196, Galaxy 0.223 vs 0.257); lr caveat
  (PC 1e-2, BP 1e-3) stands. Under occlusion F and BPf are equal after calibration.

## Correction to findings/32
The findings/32 'Gmap' rows (PC weights under BP inference) ran at the PC step scale c = 1, where BP inference
oscillates (energy rises on 37% of steps; 32% output change with batch size; dbg job 669e6547). They are unreliable.
Here the PC weights run at the BP reference's c, with a settling gate.
