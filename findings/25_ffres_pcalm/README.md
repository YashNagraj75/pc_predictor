# 25 — Residual feedforward predictor on the µPC network, depths 3 and 10 (E1)

Host job a91e2def (commit 5eb580a), finished 2026-10-20 22:11; 18/18 runs finite. Prereg: ffres_prereg.json (rev 4).
Network: jpc MLP + identity skips, µPC scaling for every rule. BP: exact gradient, lr 1e-3 fixed (user decision, not
tuned), frozen reference. PC / PC-ALM: T, alpha, lr selected per depth on a 90/10 train holdout.
Selected: PC-ALM d3 T=20 a=1 lr 3e-3; d10 T=100 a=0.25 lr 1e-2 (grid top). PC d3 T=100 lr 3e-3; d10 T=100 lr 1e-2 (grid top).

Test nmse, k=1, mean (sd) over 3 seeds:

| | BP (frozen) | plain PC | PC-ALM |
|---|---|---|---|
| d3 | 0.3603 (0.0002) | 0.3794 (0.0020) | 0.3814 (0.0018) |
| d10 | 0.3512 (0.0006) | 0.3664 (0.0018) | 0.3871 (0.0037) |

- Parity (all 3 paired |diff| <= 0.01): no arm. PC-ALM - BP: d3 +0.019..+0.022, d10 +0.032..+0.039.
- Depth (all 3 paired d10-d3 < -0.01): plain PC yes (-0.016/-0.012/-0.012); BP improves -0.009 (below margin);
  PC-ALM gets worse (+0.000/+0.009/+0.008).
- Credit (cos to BP on the same network, fixed probe): PC-ALM d3 0.986 -> 0.770 (min layer 0.99 -> 0.70);
  d10 0.95 -> 0.46 (min layer 0.96 -> 0.21). Plain PC d10 0.56 -> 0.22. "Holds (>= 0.95 throughout)": no.
- Every arm on this network is worse than the earlier plain MLP (findings/22: BP 0.299, PC-ALM 0.309). BP at lr 1e-3
  on µPC weights moves ~0.1% per Adam step; it was not tuned by design, and PC chose the largest lr at depth 10, so the
  absolute level is step-size-limited and the PC/BP comparison favours PC's tuning.
