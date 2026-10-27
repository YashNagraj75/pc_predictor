# 27 — GPU compiled-gradient audit (2026-10-25)

> **CORRECTION (2026-10-27):** the audit measures a standalone `filter_jit` of the gradient. For residual ff BP that
> standalone gradient is wrong (cos 0.23) but BP *training* was unaffected (retrain with the fix reproduced every seed,
> findings/29/bp_retrain_check). PC-ALM training WAS affected (findings/25 CORRECTION). So a failed audit flags a path
> to check, not proof that a training run is wrong; the training-level check is a retrain with the fix.

Host job bde58a8b (commit 7b12a3f), `Q1-PhaseA/audit_jit.py`. For each training path, the training gradient at TRAINED
weights is computed compiled (`eqx.filter_jit`) and uncompiled (`jax.disable_jit()`) on the host A4000 (jax 0.11.0),
once with default settings (as all past runs trained) and once with `NVIDIA_TF32_OVERRIDE=0
XLA_FLAGS=--xla_gpu_autotune_level=0`. Table: `q1_gpu_gradient_audit.csv`.

- **Only the residual feedforward µPC network (pred_ffres, findings/25) is wrong:** cos 0.234 (BP d10; also 0.47 BP d3,
  0.14-0.18 PC-ALM d10 in the earlier audit, findings/25/diag). findings/25 stays INVALID.
- **Everything else stands** (cos >= 0.9996 by default): generative sep (findings/26), stage 1 (21), C2 (23), Qi screen
  code path (pc rule, 24), plain feedforward (22 and earlier), Phase-A encoder BP / plain PC / PC-ALM.
- Minor: frozen BP auto-decoder d3 (G1, seed 1) cos 0.9948 / relmax 0.178 by default (0.99999 with the fix). Kept frozen
  (user rule); noted as a caveat on the d3 BP reference.
- With the fix every path is >= 0.99964 (C2 d10 PC-ALM is 0.9996 in both modes: intrinsic float32 sensitivity of that
  long multiplier loop, not the fault).
- All future host jobs set both variables.
