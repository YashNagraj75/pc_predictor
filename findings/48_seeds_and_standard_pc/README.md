# 48 — Seeds 1-2 of gain-matched ePC at depth 10, and the standard-PC baseline (2026-11-25)

A4000 jobs 7b2c751c (spc_u1, commit b315c0b) and 15611019 (latent pass 2 + seeds, commit 1ab7f83; exit 0, outputs
harvested by hand over Tailscale). Test nmse on 2000 CIFAR-10 test images x 40 mask shapes. Robustness columns use the
reporting rule: alpha fitted on clean data ("cleanalpha"); direction-only (alpha refit) shown as the control.

| depth 10 | clean | ctx drop 25% | ctx drop 50% | square (clean alpha / refit) | noise (clean alpha / refit) | kNN |
|---|---|---|---|---|---|---|
| BP, 5 seeds | 0.1897 ± 0.0001 | 0.2223 ± 0.0002 | 0.2784 ± 0.0003 | 0.5001 / 0.4745 | 1.1235 / 0.8414 | 0.546 |
| ePC gain-matched (epc_alm_blk4), 3 seeds | 0.2123 ± 0.0130 | 0.2492 ± 0.0141 | 0.3145 ± 0.0166 | 0.5096 / 0.4848 | 0.9734 / 0.8234 | 0.543 |
| standard PC (spc_u1), seed 0 | 0.7003 | 0.7114 | 0.7537 | 0.7027 / 0.6976 | 1.0656 / 0.8702 | 0.562 |

- Seeds 0/1/2 clean: 0.1987 / 0.2245 / 0.2136, i.e. 4.7 / 18.3 / 12.6 % above BP. Mean gap +11.9 %; seed 0 was the best seed.
  The ePC seed spread (sd 0.013) is ~100x BP's (sd 0.0001).
- Noise (clean alpha): ePC 13.4 % lower than BP on every seed (0.92-1.03 vs 1.12-1.13); direction-only 2.1 % lower.
- Context removal: BP better by 12-13 %, the same relative gap as clean.
- Depth 6 (seed 0): ePC 0.213 vs BP 0.192 (+11 %), so with seeds the gap does NOT narrow with depth.
- Standard PC, depth 6 (seed 0): clean 0.6792, noise 1.3132 raw. Its higher kNN-of-predictions (0.56 vs 0.55) is not
  explained yet; latent pass 2 (which includes spc_u1) passed all output checks and is to be read next.
