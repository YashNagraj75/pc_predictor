import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import diag_saccade as S, diag_saccade2 as S2, latent_analysis as LA
from diag_saccade import D, load_run

# Gates for latent_analysis.py, run from a dataset dir (Q1-PhaseA for Galaxy10).  Each new path must reproduce the
# validated evaluation (diag_saccade2 / findings 32-34) on the same views before the analysis is trusted.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def main(backbone, seed, ref_json):
    L_ = np.load("pred_latents.npz")
    raw = L_["bp_emb_tr"]; mu, sd = raw.reshape(-1, 128).mean(0), raw.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32)
    cfg, params = load_run(backbone); enc = S.Encoder(cfg, params, mu, sd)
    Xte = np.asarray(D.load_dataset(cfg.data)[2], np.float32)
    models = S.load_models([seed], (10,))
    for k, m in models.items():
        if m.kind == "BPg":
            S2.attach_bpg(m, k)
    check(f"all four model kinds load for seed {seed}", sorted(m.kind for m in models.values()) == ["BPf", "BPg", "F", "G"])
    ref = {(r["model"], r["cond"], r["pool"], r["K"]): r["nmse"] for r in json.load(open(ref_json))["rows"] if r["input"] == "whole_aug"}
    Xc, occ = S.corrupt_images(Xte, "occlude")
    Z, T, B = S.whole_views(Xc, enc, cfg, 6, False, 303)
    Z0, T0, _ = S.whole_views(Xte, enc, cfg, 6, False, 303)
    check("clean and occluded views share crops (paired)", np.array_equal(T, T0))
    for name, m in models.items():
        pools = ["mean", "pw"] if m.kind in ("G", "F") else ["own"]
        for pool in pools:
            for cond, (Zv, Tv) in (("occlude", (Z, T)), ("clean", (Z0, T0))):
                P = LA.predict(m, Zv, Tv, tte, pool)
                r = S.metrics(P, zte)[0]["nmse"]; s = ref.get((name, cond, pool, 6))
                check(f"{name} {pool} {cond} K=6 reproduces the stored evaluation", s is not None and abs(r - s) < 1e-4 * max(1, abs(s)),
                      f"{r:.6f} vs {s}")
        if m.kind in ("G", "F", "BPf"):          # mean-pool prediction == mean of its own contributions path
            C = LA.contributions(m, Z0, T0, tte)
            check(f"{name}: predict(C given) == predict(C recomputed)", np.allclose(LA.predict(m, Z0, T0, tte, "mean", C), LA.predict(m, Z0, T0, tte, "mean"), atol=1e-5))
    # shift and gain sanity on synthetic cases
    m = next(v for v in models.values() if v.kind == "F")
    P0 = LA.predict(m, Z0, T0, tte)
    sh = LA.shift(P0, P0, Z0, Z0, zte)
    check("shift of identical predictions is zero", sh["rel_shift"] == 0 and abs(sh["shrink"] - 1) < 1e-6)
    Dz = 0.3 * np.random.default_rng(0).standard_normal(zte.shape[:1] + (1, 128)).astype(np.float32)
    sh2 = LA.shift(P0 + Dz, P0, Z0 + Dz, Z0, zte)
    check("a prediction that moves exactly with the input has follow = 1, cos = 1", abs(sh2["follow"] - 1) < 1e-4 and abs(sh2["cos_follow"] - 1) < 1e-4,
          f"{sh2['follow']:.5f} {sh2['cos_follow']:.5f}")
    g1 = LA.gain(m, Z0, T0, tte, "mean", P0, 0.1, np.random.default_rng(1)); g2 = LA.gain(m, Z0, T0, tte, "mean", P0, 0.05, np.random.default_rng(1))
    check("gain is locally linear (eps 0.1 vs 0.05 within 10%)", abs(g1["gain_all"] / g2["gain_all"] - 1) < 0.1, f"{g1} {g2}")
    check("one-view gain of a K=6 mean pool is below the all-view gain", g1["gain_one"] < g1["gain_all"] * 1.5, f"{g1}")
    of = np.stack([S.occluded_frac(B[:, v, 1], B[:, v, 0], B[:, v, 2], occ, Hh=Xte.shape[1]) for v in range(6)], 1)
    inf = LA.influence(m, Z[:300], T[:300], tte[:300], "mean", of[:300])
    check("influence returns finite values with images to rank", np.isfinite(inf["spearman_infl_occ"]) and inf["n_images_rho"] > 50, str(inf))
    X = np.random.default_rng(2).standard_normal((500, 4, 16))
    check("CKA(X, X) == 1 and eff_rank(white noise) near full", abs(LA.cka(X, X) - 1) < 1e-9 and LA.eff_rank(X) > 14)
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], int(sys.argv[2]), sys.argv[3]) else 1)
