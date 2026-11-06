# 40 — Stage A, depth 10 (in progress): ePC-ALM vs BP I-JEPA predictor on the frozen V0 encoder

> **INVALIDATED 2026-11-06 (findings/41):** trained and/or credit-measured with XLA GPU autotuning ON, which gives wrong
> batched gradients (rel error 0.46-0.61 vs CPU). Evaluation is correct, the training gradients were not. Rerun pending.

Host jobs 936a85c5 + deeccfb8 (commit 421f84f), seed 0, 25 epochs, test = 2000 CIFAR-10 test images x 40 mask shapes.
Both arms selected lr 3e-3 (grid EDGE) on the 90/10 split (5-epoch val: BP 0.1982, ePC-ALM 0.2013).

| | clean | ctx drop 25% | 50% | square (cal) | noise (cal) | kNN of predictions |
|---|---|---|---|---|---|---|
| BP lr 3e-3 | 0.1898 | 0.2223 | 0.2784 | 0.4741 | 0.8420 | 0.5425 |
| ePC-ALM rho = gain/4, T 10 | 0.2028 | 0.2382 | 0.2997 | 0.4794 | 0.8456 | 0.5435 |
| gap | +0.0130 | +0.0159 | +0.0213 | +0.0054 | +0.0037 | +0.0010 |

- Clean gap 0.013 exceeds the pre-registered parity margin of 0.01: BP is better on clean and context-drop prediction.
  ePC-ALM is about as good as BP at lr 3e-4 (0.1997). Square and noise (calibrated) gaps are within 0.006; kNN is equal.
  The paired per-image bootstrap has not been computed yet (per_image.npz still on the host).
- Credit drifts during training: cos to BP 0.96 (step 1) -> 0.95 (200) -> 0.85 (800) -> ~0.82-0.90 (1400-3200)
  -> 0.41-0.44 (4600-4875); size ratio falls to 0.43; at the end the median leaf is 0.43, the weakest -0.31.
- Cause, as logged: the gain spread across blocks is very wide (median 2-9e3, max 2e4-1.2e5), so rho = median gain / 4 is far
  below the gain of the high-gain blocks, the same failure as the fixed-rho gates (findings/39) and the generative decoder
  (findings/37).
- Step 1000 shows a transient loss spike (0.41) with the gain collapsing to 0.08; it recovered by step 1400. Not explained yet.
- Cost: 34,960 s for the 25-epoch ePC-ALM run vs about 3,570 s for BP (about 9.8x).
