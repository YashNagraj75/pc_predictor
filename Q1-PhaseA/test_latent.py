import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax.numpy as jnp, pred_gpc as G
import diag_saccade as S, diag_saccade2 as S2, latent_analysis as LA
from diag_saccade import D, load_run

# Gates for latent_analysis.py (rev 2), run from a dataset dir.  (1) every analysed model reproduces the stored evaluation
# on the same views, for every seed; (2) the swap models are what they claim; (3) known-answer tests for every statistic.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


class Lin:                                       # linear stub: per-view prediction A z, mean-pooled (known answers)
    kind = "F"

    def __init__(self, A):
        self.A = A


def stub_contrib(m, Z, T, tte):
    return np.repeat((Z @ m.A.T)[:, :, None], tte.shape[1], 2)


def known_answers():
    rng = np.random.default_rng(0); n, K, d = 400, 6, 16
    z = rng.standard_normal((n, 4, d)); P0 = z + 0.3 * rng.standard_normal(z.shape)
    g = LA.geometry(2 * P0, z); g0 = LA.geometry(P0, z)
    check("calibrated nmse and cosine are scale-invariant; alpha* halves when P doubles",
          abs(g["nmse_calibrated"] - g0["nmse_calibrated"]) < 1e-9 and abs(g["alpha_star"] - g0["alpha_star"] / 2) < 1e-9)
    Z0 = rng.standard_normal((n, K, d)); Dz = rng.standard_normal((n, 1, d))
    sh = LA.shift(P0 + 0.5 * Dz, P0, Z0 + Dz, Z0, z)
    check("shift: P moving by 0.5 x the input shift has gain_real 0.5, and the nmse change decomposes exactly",
          abs(sh["gain_real"] - 0.5) < 1e-9 and abs(sh["d_nmse"] - sh["d_nmse_shift_term"] - sh["d_nmse_cross_term"]) < 1e-9,
          f"{sh['gain_real']:.4f}")
    shr = LA.shift(0.7 * P0, P0, Z0 + Dz, Z0, z)
    check("shift: pure shrinkage is all radial with negative radial mean", shr["frac_radial"] > 0.999 and shr["radial_mean"] < 0)
    check("shift fractions sum to 1", abs(sh["frac_radial"] + sh["frac_along_input"] + sh["frac_orth"] - 1) < 1e-9)
    # gain of a linear mean pool: coherent = ||A u|| / ||u|| ~ ||A||_F / sqrt(d) for random u; independent the same per unit mean
    A = rng.standard_normal((d, d)) / np.sqrt(d); m = Lin(A); t = np.zeros((n, 4, 9), np.float32)
    orig_c, orig_p = LA.contributions, LA.predict
    LA.contributions = lambda mm, Z, T, tte, _c=orig_c: stub_contrib(mm, Z, T, tte) if isinstance(mm, Lin) else _c(mm, Z, T, tte)
    LA.predict = lambda mm, Z, T, tte, pool="mean", C=None, _p=orig_p: (LA.pool_c(mm, LA.contributions(mm, Z, T, tte), tte)
                                                                         if isinstance(mm, Lin) else _p(mm, Z, T, tte, pool, C))
    gg = LA.gain(m, Z0.astype(np.float32), np.zeros((n, K, 9), np.float32), t, "mean", 0.1)
    ref = np.linalg.norm(A) / np.sqrt(d)
    check("gain of a linear mean pool: coherent, independent and one-view all equal ||A||_F/sqrt(d) per unit mean perturbation",
          all(abs(gg[k] / ref - 1) < 0.1 for k in ("gain_coherent", "gain_indep", "gain_one")), f"{gg} ref {ref:.3f}")
    # influence: a view whose contribution is scaled up by its displacement has the largest influence; rho = +1 on disp
    disp = np.tile(np.arange(K, dtype=float), (n, 1)) + 0.01 * rng.standard_normal((n, K))
    Zs = Z0 * (1 + disp[..., None])
    inf = LA.influence(m, Zs.astype(np.float32), np.zeros((n, K, 9), np.float32), t, disp, disp)
    check("influence: rho(influence, displacement) near +1 when displacement scales the contribution", inf["rho_infl_disp"] > 0.8, str(inf))
    LA.contributions, LA.predict = orig_c, orig_p
    X = rng.standard_normal((500, 4, 16))
    check("CKA(X, X) == 1 and eff_rank(white noise) near full", abs(LA.cka(X, X) - 1) < 1e-9 and LA.eff_rank(X) > 14)


def main(backbone, seeds, ref_json):
    known_answers()
    L_ = np.load("pred_latents.npz")
    raw = L_["bp_emb_tr"]; mu, sd = raw.reshape(-1, 128).mean(0), raw.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32)
    cfg, params = load_run(backbone); enc = S.Encoder(cfg, params, mu, sd)
    Xte = np.asarray(D.load_dataset(cfg.data)[2], np.float32)
    models = LA.build_models(seeds)
    kinds = sorted({m.kind for m in models.values()})
    check(f"model kinds load for seeds {seeds}", {"G", "F", "BPf", "Gmap", "MapPool"} <= set(kinds), str(kinds))
    ref = {(r["model"], r["cond"], r["pool"], r["K"]): r["nmse"] for r in json.load(open(ref_json))["rows"] if r["input"] == "whole_aug"}
    Xc, occ = S.corrupt_images(Xte, "occlude")
    Z, T, B = S.whole_views(Xc, enc, cfg, 6, False, 303)
    Z0, T0, B0 = S.whole_views(Xte, enc, cfg, 6, False, 303)
    check("clean and occluded views share crops (paired)", np.array_equal(T, T0) and np.array_equal(B, B0))
    for name, m in sorted(models.items()):
        if m.kind == "MapPool":
            continue
        pools = ["mean", "pw"] if m.kind in ("G", "F") else ["own"]
        for pool in pools:
            for cond, (Zv, Tv) in (("occlude", (Z, T)), ("clean", (Z0, T0))):
                P = LA.predict(m, Zv, Tv, tte, pool); r = S.metrics(P, zte)[0]["nmse"]; s = ref.get((name, cond, pool, 6))
                if m.kind == "Gmap":                 # findings/32 ran Gmap at the unstable c = 1; nothing to reproduce
                    print(f"     report: {name} {cond} K=6 nmse {r:.4f} (no stored value)"); continue
                check(f"{name} {pool} {cond} K=6 reproduces the stored evaluation", s is not None and abs(r - s) < 1e-4 * max(1, abs(s)), f"{r:.6f} vs {s}")
        if m.kind in ("G", "F", "BPf"):
            C = LA.contributions(m, Z0, T0, tte)
            check(f"{name}: mean of contributions == predict", np.allclose(LA.pool_c(m, C, tte), LA.predict(m, Z0, T0, tte, "mean"), atol=1e-5))
    # swap models: a 1-view MapPool equals the 1-view joint MAP of the same weights
    for name, m in sorted(models.items()):
        if m.kind != "MapPool":
            continue
        joint = models[name.replace("BPgMP_", "BPg_").replace("GMP_", "Gmap_")]
        z, t, tt = Z0[:300, :1], T0[:300, :1], tte[:300]
        P1 = LA.pool_c(m, m.beliefs(z, t, chunk=250), tt); J1 = S2._bp_predict(joint, z, t, tt, 1, batch=250)
        e = float(np.max(np.abs(P1 - J1)) / np.max(np.abs(J1)))
        check(f"{name}: at K=1, mean of per-view MAP == joint MAP of the same weights (same batching)", e < 1e-4, f"rel {e:.1e}")
        J3 = S2._bp_predict(joint, z, t, tt, 1, batch=300); eb = float(np.max(np.abs(J3 - J1)) / np.max(np.abs(J1)))
        hist = np.asarray(G.infer_s(joint.p, None, jnp.asarray(Z0[:300, :6]), jnp.asarray(T0[:300, :6]), joint.cfg)[2])
        mono = float(np.mean(np.diff(hist) <= 1e-6 * np.abs(hist[:-1])))
        check(f"{name.replace('MP_', '_')}: BP inference settles (energy non-increasing on >= 99% of steps at K=6; batch-size effect < 1e-2)",
              mono >= 0.99 and eb < 1e-2, f"monotone {mono:.3f}, batch effect {eb:.1e}, c {joint.cfg.c}")
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]) else 1)
