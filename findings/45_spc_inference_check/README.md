# 45 — Standard PC (spc_u1) on the ViT predictor: inference-only credit check before training (2026-11-12)

Jobs 12273ce4 (float32, 64 test-split train images) and 1302cff3 (float64, 16 images), `Q1-PhaseA/diag_spc.py`, trained BP
checkpoints (d6, d10, lr 3e-3, seed 0) and an untrained d10 predictor. Standard PC = relaxation over the activities from the
feedforward pass, step 0.9 / lambda_max(Hessian), no multipliers. Compared with the BP gradient on the same batch.

- rho = 1, T = 20 (the training setting): cosine to BP 0.53-0.54 (f32) / 0.61-0.62 (f64) on trained nets, gradient size 0.41-0.46.
  Error reaches only the top 6 of 12 (d6) / 7 of 20 (d10) nodes above 1e-3 of the top error; 57% (d6) / 69% (d10) of
  parameter leaves get under 1e-3 of BP's gradient size.
- More steps do not approach BP: rho = 1, T = 80 / 320 -> cosine 0.40 / 0.30 (trained d10); the relaxation reaches every node but
  settles where the hidden errors absorb the output error.
- rho = gain/4 helps a little (cosine 0.67-0.70) with the same reach limit at T = 20.
- float32 vs float64: same reach and same starved-leaf fraction. In float64 the bottom nodes' errors are 1e-15..5e-8 of the
  top error (geometric decay, one node per step); float32 replaces them by a ~6e-7 rounding floor. So the starvation is a
  property of the method at this T, not a precision artefact; float32 only adds noise where the true signal is negligible.
- Step-size estimate: 4 power iterations underestimate lambda_max by up to 54% (unsafe); 8 by 1.5-4%; 20 by < 1%.
  The training run uses 8 (user decision).
