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
## Cause and fix: Triton GEMMs (flag sweep, job 49083aaf; flag_sweep_vs_cpu.txt)
Real training shapes (batch 64; trained d10 predictor; ePC chunk 32, T 10, 4 its), each GPU setting vs a CPU reference:

| setting | pred BP rel / time | pred ePC rel / time | V0 loss (fresh init) rel |
|---|---|---|---|
| default (autotune on, what we trained with) | 0.93 (cos 0.37) / 0.157 s | 0.88 (cos 0.48) / 2.37 s | 2e-7 |
| autotune off | 8e-7 / 0.678 s | 3e-6 / 8.63 s | 2e-7 |
| **Triton GEMM off (autotune on)** | **1e-6 / 0.164 s** | **3e-6 / 2.24 s** | 2e-7 |
| cuBLASLt off, level 2, deterministic ops | 0.88-0.93 wrong | wrong | 2e-7 |

- Triton GEMM kernels are the cause; disabling only them is correct and as fast as the faulty default.
- The V0 loss gradient was correct under every setting at its init weights, so V0 (findings/38) may be valid; the
  gate job re-checks it at batch 256 with the trained encoders.
- Fix (commit 504e742): ViT jobs use XLA_FLAGS=--xla_gpu_enable_triton_gemm=false; gpu_gate.py compares GPU and CPU
  gradients at the run's shapes and trained weights before every depth/noise/a0 job (rel < 1e-4 or the job stops).

## The fault is intermittent (job 610dccdd, 2026-11-06)
Under the DEFAULT flags, a fresh process at batch 64 (BP checkpoint bp_d10_lr1e-3) gave CORRECT gradients for BP, ePC and
per-block ePC (rel <= 2e-6 vs CPU), while the sweep (ALM checkpoint, same shapes) and job 23b9a59a (both checkpoints,
batch 4) gave 46-93% errors. Autotuning picks kernels by timing at compile time, so whether a faulty Triton GEMM is chosen
can vary between processes. Consequences: a gate run under the default flags can pass by luck, so the gate is only
meaningful together with Triton off (cuBLAS path); and the V0 encoder, whose loss gradient was correct in every one of
9 default/variant compilations at batch 64 and 256, is probably but not provably unaffected.
The full ViT gate suite passes under Triton off (54/54).
