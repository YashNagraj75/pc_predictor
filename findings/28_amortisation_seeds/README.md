# 28 — Amortisation vs multipliers, generative predictor, 3 seeds (GPU fix on)

Host jobs 1b729766 (9 runs, hit its 20 h limit) + cbd08078 (resume, commit 12b318f), 2026-10-25/27. All 18 runs passed,
audit gate cos 1.0000. Settings frozen from the E2/E3 selection: PC-ALM rho 16, k_in 1, T2 200; no-multiplier arms rho 1.
BP reference = frozen auto-decoder G1 (not retrained): d3 0.3053 (0.0008), d10 0.2783 (0.0008).

Test nmse, one view seen, mean (sd) over seeds 0-2:

| arm | d3 | d10 | d10 - d3 (per seed) |
|---|---|---|---|
| amortised + PC-ALM | 0.3309 (0.0009) | **0.3075 (0.0018)** | -0.025 / -0.025 / -0.020 |
| amortised, no multipliers | 0.3989 (0.0015) | 0.5133 (0.0129) | +0.100 / +0.123 / +0.120 |
| plain-PC belief, no multipliers | 0.3815 (0.0005) | 0.3753 (0.0007) | -0.007 / -0.006 / -0.005 |

- Multipliers (amortised + PC-ALM vs amortised alone): -0.068 at d3, -0.19 to -0.22 at d10, every seed.
- Amortisation alone (vs plain-PC belief, no multipliers): HURTS, +0.017 at d3, +0.12 to +0.15 at d10.
- The combination beats the plain-PC belief by 0.050 (d3) and 0.068 (d10).
- Depth: amortised + PC-ALM improves d3 -> d10 by 0.020-0.025 on every seed (> 0.01 margin): scales with depth.
  Plain PC: -0.006 (below the 0.01 margin). Amortised alone gets much worse.
- No parity with BP: gap 0.026 (d3) and 0.029 (d10); BP improves with depth by 0.027, about the same.
- Credit (cos to the BP auto-decoder gradient, end of training): 0.96 (d3), 0.71 (d10) with multipliers; 0.73 / 0.27-0.31 without.
- Robustness, d10: noise 0.2 PC-ALM 1.797 vs BP 1.887 (PC better); occlusion 2.156 vs 2.053 (BP better). d3: BP better on both.
