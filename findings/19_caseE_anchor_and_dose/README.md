# 19 — Case E (reconstruction anchor) and the PC dose measurement — CPU only, no GPU run

2026-10-08, seed 0, 160 CPU steps, n = 256 selection rows. Not a pilot: these are
smoke-scale results, reported because they decided not to spend the GPU pilot.

1. Three reviews (reviewE_*.txt) found the first case-E code's sparse solver did not
   converge (verified: stacked Lipschitz 15 -> ~400; 17-step code 66 % from the LASSO
   optimum, 56 % active vs 29 % at the optimum) and calibration on clean rather than
   augmented views. Fixed (code_k = 8, convergence gate, brightness-removed target,
   augmented-view calibration, D@0.1x control). A proposed band widening was withdrawn.
2. After the fix the codes converge (7-9 % from optimum, ~25 % active, pixel R^2 0.73),
   but PC's layer-5 effective rank falls 7.0 -> ~2.0 with the anchor and ~1.7-1.9
   without it; BP holds (4.5 with anchor, 6.5 without). PC with NO heads: 7.2 -> 7.9.
3. diag_dose.py: at init, PC's head/task gradient ratio is 365 (layer 0), 98, 29,
   6.4, 1.6, 0.45 (layer 5) where BP's is 0.10 at every layer. At T = 16 PC's task
   credit barely reaches the bottom layers, so the head term trains them. Matching
   PC's aggregate dose to 0.1x (heads x0.13 K=1, x0.40 K=5) still collapses PC
   (rank ~2.0), since layers 0-3 stay head-dominated (47x .. 3.7x).
