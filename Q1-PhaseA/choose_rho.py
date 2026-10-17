import json, sys
import numpy as np

# Pre-registered (gpc_alm_prereg.json, rev 2) choice of the high penalty and the phase-2 budget, per depth, from
# diag_c2.json.  m(T2) = median over scale in {1, 2}, k in {1, 3}, c in {0.5, 1} of the all-view credit cosine.
# rho_hi = smallest rho in {4, 16, 64} with m(100) >= 0.95 (T2 = 100); else smallest with m(400) >= 0.95 (T2 = 400);
# else rho = 64, T2 = 400, flagged.  A rho with any non-finite row at scale <= 2 is not eligible.
rows = json.load(open(sys.argv[1])); out = sys.argv[2]
choice = {}
for d in (3, 10):
    med = {}
    for rho in (4.0, 16.0, 64.0):
        R = [r for r in rows if r["depth"] == d and r["rho"] == rho and r["scale"] <= 2.0 and r["c"] in (0.5, 1.0)]
        if not R or not all(r["finite"] and r.get("finite_T2000", False) for r in R):
            continue
        med[rho] = {T2: float(np.median([r[f"cos_all_T2_{T2}"] for r in R])) for T2 in (100, 400, 1600)}
    pick = next(((rho, 100) for rho in sorted(med) if med[rho][100] >= 0.95), None) \
        or next(((rho, 400) for rho in sorted(med) if med[rho][400] >= 0.95), None)
    flag = pick is None
    rho, T2 = pick or (64.0, 400)
    choice[str(d)] = {"rho_hi": rho, "T2": T2, "credit_bar_not_met": flag, "median_cos": {str(k): v for k, v in med.items()}}
json.dump(choice, open(out, "w"), indent=1)
print(json.dumps(choice))
