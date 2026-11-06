# 41 — XLA GPU autotuning gives wrong ViT training gradients (forward is correct)

Diagnostic host job 23b9a59a (worktree at 9d9618a), 2026-11-05/06. Trained depth-10 stage-A predictors (ePC-ALM and BP
checkpoints of findings/40), 4 images x 4 target blocks; batched gradients (chunk 4) vs the weighted sum of one-block
gradients, and batched gradients against a CPU reference.

| batched gradient | GPU autotune ON vs CPU | GPU autotune OFF vs CPU |
|---|---|---|
| BP, ALM checkpoint | rel 0.61, cos 0.793 | rel 2.2e-6 |
| ePC alpha 0, rho 500, ALM checkpoint | rel 0.46, cos 0.887 | rel 1.7e-6 |
| ePC alpha 1 / BP checkpoint (on vs off) | rel 0.48-0.60, cos 0.81-0.88 | — |

- Forward predictions agree batched vs single to 7e-7 under both settings: evaluations are correct.
- Autotune ON was re-enabled for ViT jobs in run 6 (commit 648861f) after compiled==eager audits passed on small or
  untrained networks; those audits did not cover these shapes and weights. Same failure class as findings/27.
- Affected: every ViT training run so far was trained with wrong gradients (V0 encoder, findings/38; all stage-A runs,
  findings/39-40), and every in-training or A0 credit number (cos to BP) was measured under the fault.
  findings/38-40 are therefore INVALID as learning-rule comparisons until rerun; the V0 gate numbers are correct
  measurements of an encoder trained with corrupted gradients.
- Next: flag sweep at real training shapes vs CPU (job 49083aaf) to find a setting that is correct and fast.
