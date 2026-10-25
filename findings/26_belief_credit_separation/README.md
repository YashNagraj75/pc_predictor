# 26 — Generative PC-ALM: belief separated from credit (E2/E3), seed-0 screen

> **AUDITED (2026-10-25, findings/27):** every sep training path used here trained on correct gradients (compiled ==
> uncompiled, cos 1.0000). Seed 0 only; seeds 0-2 rerun with the numerics fix in job_sep_seeds.sh.


Host job 0866568b (commit 369a7f8), finished 2026-10-21 05:14; 12/12 runs, none diverged, no step retries.
Prereg: sep_prereg.json. Selected: PC-ALM rho 16, k_in 1, T2 200 (both depths); no-multiplier controls rho 1.
Reference: frozen BP auto-decoder G1, 3-seed mean d3 0.3053, d10 0.2783 (sd 0.0008).

Test nmse, k=1 seen view (amortised beliefs are one pass, so 'settled' equals k1):

| belief | multipliers | d3 | d10 |
|---|---|---|---|
| oracle (BP-AD inference) | PC-ALM | 0.3205 | 0.3775 |
| oracle | none | 0.3837 | 0.3983 |
| plain-PC relaxation | PC-ALM | 0.3670 | 0.8355 (settled 1.63) |
| plain-PC relaxation | none | 0.3815 | 0.3747 |
| amortised | PC-ALM | 0.3285 | **0.3077** |
| amortised | none | 0.3990 | 0.5007 |

- Best generative PC result so far: amortised + PC-ALM at d10, 0.308 (C2 best d3 0.352; plain PC 0.376-0.381). It is
  the only PC arm that improves with depth (0.329 -> 0.308) but is still 0.029 behind BP at d10 (0.023 at d3 for oracle).
- Multipliers help with the oracle and amortised beliefs at both depths (0.06-0.19 nmse); with the PC belief they help
  little at d3 and collapse d10.
- Oracle + PC-ALM is stable at d10 (finite, res_rel 0.0025) but worse than at d3; its belief norm at the end is 0.07.
- Credit (cos to the BP-AD gradient): ~0.96-0.99 at d3 throughout; at d10 it ends 0.63 (oracle), 0.77 (amort), 0.74 (pc).
- Caveats: one seed; 'oracle' is not a local method; the amortiser's target uses a dL/ds that the float64 gate found
  5-12% (d3) and 27-62% (d10) off (sep_prereg caveat_g_s). Under the oracle test belief the amortised d10 decoder
  scores 0.336.
