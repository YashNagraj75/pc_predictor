# 38 — V0: backprop I-JEPA ViT-Tiny encoder on CIFAR-10 (frozen for stage A)

> **INVALIDATED 2026-11-06 (findings/41):** trained and/or credit-measured with XLA GPU autotuning ON, which gives wrong
> batched gradients (rel error 0.46-0.61 vs CPU). Evaluation is correct, the training gradients were not. Rerun pending.

Host job 24d31ccf (commit 648861f), 2026-11-03 10:54-14:24 host time; 54/54 gates; 300 epochs (58,500 steps, ~0.21 s/step).
ViT-Tiny depth 6, width 192, patch 4, I-JEPA multi-block masking, EMA target encoder. Gate fixed before the run, on the
full 50k train / 10k test sets, mean-pooled target-encoder features: centred kNN >= 0.4817, linear >= 0.55, eff. rank >= 19.

| encoder | kNN | kNN centred | linear | eff. rank |
|---|---|---|---|---|
| **trained (V0)** | **0.671** | **0.669** | **0.722** | **42.5** |
| random init (3 seeds) | 0.299-0.304 | 0.287-0.293 | 0.347-0.359 | 2.5-2.6 |
| raw pixels | 0.429 | 0.432 | 0.399 | - |

GATE PASS. Caveat: the in-training kNN (512 test images) peaked at 0.632 near step 29k and drifted to 0.616 by the end,
while the positional share of target features rose from 0.25 to 0.37. Mild late positional drift, not collapse
(rank stays ~40). Weights: host runs_vit/v0_d6_s0/{context,target}_encoder.eqx (sha256 in v0_report.json).
