# 37 — ePC (Goemaere et al., ICML 2026) vs state PC: inference-only credit check on trained depth-10 checkpoints

Host job abe904bb (commit 4d310b0; not in the compute ledger, harvested by hand), 2026-11-02. 12/12 gates (convex
known-answer tests on linear networks; ePC at large precision == BP). Credit = cosine (and norm ratio) of the PC weight
gradient to BP's on the same weights and batch. Tables: q1_epc_ff_credit.csv, q1_epc_gen_credit.csv.

## Feedforward (residual PC-ALM rho16 d10, 3 seeds; plus the lr-matched BP checkpoint)
- ePC settles within 20 steps (values flat from T = 20 to 2000).
- ePC-ALM rho 64, T 20: cos 0.995 (weakest layer 0.990, ratio 1.00); on BP-trained weights 0.987 (ratio 1.00).
- ePC-ALM rho 16, T 20: 0.976. ePC without multipliers needs rho ~1024 (0.994, ratio 0.93).
- State PC-ALM rho 16 (our training setting): 0.31 at T 20, 0.88 at T 400, ratio unstable 0.27-1.39.
-> ePC removes the slow-inference problem of the feedforward predictor.

## Generative (plain-PC-trained d10 seeds 1,2; BP auto-decoder d10 seeds 1,2). Fixed-belief credit vs the BP auto-decoder
- ePC is stable (energy never rises) where state PC at hidden precision 16 diverges on the PC-trained decoders.
- Credit is limited by the EQUILIBRIUM, not by inference speed: values are flat in T but low unless the hidden
  precision approaches the decoder's hidden-error gain lambda_max(J_e^T J_e).
  - PC-trained decoders: gain ~5.0e4; credit cos 0.05 (precision 1) -> 0.48 (256) -> 0.80 (4096), ratio only 0.22.
  - BP-trained decoders: gain ~1.1-1.4e4; credit 0.49 -> 0.79 -> 0.92 (4096), ratio 0.90.
- At low precision the hidden errors explain the observation almost exactly (observation nmse -> 0.000 on PC decoders),
  leaving no error to train the weights or move the belief.
- PC training produced a decoder with ~4x the hidden-error gain of BP training on the same architecture.
-> ePC cannot fix generative PC at moderate precision; it reaches BP's credit only at precision >~ the gain (1e4-5e4),
   where it is effectively the BP auto-decoder.
