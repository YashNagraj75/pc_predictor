import argparse, dataclasses as dc, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import diag_saccade as S, diag_saccade2 as S2, pred_gpc as G
from diag_saccade import D, load_run
from scipy.stats import spearmanr

# Latent analysis (Phase C, inference only), rev 2 after 3 reviews.  Why does the generative PC predictor (G) degrade less
# than its frozen BP auto-decoder (BPg) under occlusion once it sees several views, and why does the residual ff PC
# predictor (F) beat BPf on clean/noise?  Run from a dataset dir.  Views from diag_saccade2 (key 303; clean and corrupted
# views share crops, so every comparison is paired).  The stored G-BPg occlusion gap is complete by K = 6 on all datasets.
#  SWAP  weights {G, BPg} x pooling {joint MAP over K views, mean of per-view MAP beliefs, oracle-weighted mean}, plus G's
#        own amortised mean: assigns the gap to pooling vs weights (review S1)
#  A  nmse, calibrated nmse (best scalar rescale alpha*), alpha*, norm ratio, cosine (S3)
#  B  paired corruption shift D = P_c - P_0: exact nmse-change decomposition, D split radial / along input / orthogonal,
#     gain along the real corruption sqrt(rel_shift / input_rel_shift)
#  C  gain to coherent (same direction in every view) and independent seen-latent perturbations, paired across models
#  D  leave-one-view-out influence, normalised by the view's deviation from the pool, vs the view's actual latent
#     displacement and occluded fraction; centred contribution ratio; oracle pool (weights 1 - occ) as the bound on
#     what down-weighting could buy (S4, S5)
#  E  effective rank, linear CKA with targets; predictions saved (float32) for cross-model CKA
KS = (1, 6, 48)
N_GAIN = 500


def finite_rows(P):                              # images whose every predicted value is finite
    return np.all(np.isfinite(P.reshape(len(P), -1)), 1)


def eff_rank(P):
    X = P.reshape(-1, P.shape[-1]).astype(np.float64); X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2; p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def cka(A, B):                                   # linear CKA over all (image, target) rows
    A = A.reshape(-1, A.shape[-1]).astype(np.float64); B = B.reshape(-1, B.shape[-1]).astype(np.float64)
    A, B = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


# Per-view MAP beliefs pooled outside inference: weights p, BP inference (zero start) on each view alone, then a
# (weighted) mean of the K beliefs, then one top-down readout.  With G's weights the config is GMap's (review S1).
class MapPool:
    def __init__(self, p, cfg, kind, seed):
        self.p, self.cfg, self.kind, self.seed = p, cfg, kind, seed
        self._s = eqx.filter_jit(lambda p, Z, T: G.infer_s(p, None, Z, T, cfg)[0])
        self._d = eqx.filter_jit(lambda p, s, t: G.topdown(p, s, G.phi(t), cfg.H)[1])

    def beliefs(self, Z, T, chunk=4000):         # (n, K, d), (n, K, 9) -> (n, K, M), every view inferred alone
        n, K = Z.shape[:2]; Zr, Tr = Z.reshape(n * K, 1, -1), T.reshape(n * K, 1, -1)
        return np.concatenate([np.asarray(self._s(self.p, jnp.asarray(Zr[i:i + chunk]), jnp.asarray(Tr[i:i + chunk])))
                               for i in range(0, n * K, chunk)]).reshape(n, K, -1)

    def readout(self, s, t, batch=250):
        return np.concatenate([np.asarray(self._d(self.p, jnp.asarray(s[i:i + batch]), jnp.asarray(t[i:i + batch]))) for i in range(0, len(s), batch)])


def contributions(m, Z, T, tte):                 # per-view contribution before pooling
    if m.kind == "G":
        return m.embed(Z, T)                     # (n, K, M) amortised embedding
    if m.kind == "MapPool":
        return m.beliefs(Z, T)                   # (n, K, M) per-view MAP belief
    if m.kind == "F":
        return S2.ff(m, Z, T, tte)               # (n, K, 4, d)
    if m.kind == "BPf":
        return np.stack([S2._bp_predict(m, Z[:, v:v + 1], T[:, v:v + 1], tte, 1) for v in range(Z.shape[1])], 1)
    return None                                  # BPg, Gmap: joint inference, no per-view contribution


def pool_c(m, C, tte, w=None):                   # (weighted) mean of contributions -> prediction
    w = np.ones(C.shape[:2], np.float32) if w is None else w
    sh = (-1,) + (1,) * (C.ndim - 2)
    c = np.einsum("nk,nk...->n...", w, C) / np.maximum(w.sum(1), 1e-6).reshape(sh)
    return m.readout(c, tte) if m.kind in ("G", "MapPool") else c


def predict(m, Z, T, tte, pool="mean", C=None):
    K = Z.shape[1]
    if m.kind in ("BPg", "Gmap"):
        return S2._bp_predict(m, Z, T, tte, K)
    if pool == "pw" and K > 1:
        lp = S2.LPool(m, tte)
        for v in range(K):                       # one view at a time, own group, exactly as diag_saccade2.whole_pools
            lp.add(Z[:, v:v + 1], T[:, v:v + 1], -2 - v)
        return lp.predict("pw")[0]
    return pool_c(m, contributions(m, Z, T, tte) if C is None else C, tte)


def geometry(P, z):
    ok = finite_rows(P); P, z = P[ok].astype(np.float64), z[ok].astype(np.float64)
    pz, pp, zz = np.sum(P * z), np.sum(P * P), np.sum(z * z)
    nP, nz = np.linalg.norm(P, axis=-1), np.linalg.norm(z, axis=-1)
    return {"n_nonfinite": int((~ok).sum()), "nmse_check": float(np.sum((P - z) ** 2) / zz),
            "alpha_star": float(pz / pp), "nmse_calibrated": float(1 - pz ** 2 / (pp * zz)),
            "norm_ratio": float(nP.mean() / nz.mean()), "cos": float(np.mean(np.sum(P * z, -1) / (nP * nz + 1e-12)))}


def shift(Pc, P0, Zc, Z0, z):
    ok = finite_rows(Pc) & finite_rows(P0)
    Pc, P0, z = Pc[ok].astype(np.float64), P0[ok].astype(np.float64), z[ok].astype(np.float64)
    Din = np.broadcast_to((Zc[ok].mean(1) - Z0[ok].mean(1))[:, None], P0.shape).astype(np.float64)
    Dl = Pc - P0; zz = np.sum(z * z)
    u0 = P0 / (np.linalg.norm(P0, axis=-1, keepdims=True) + 1e-12)
    rad = np.sum(Dl * u0, -1, keepdims=True) * u0                       # radial part (along the clean prediction)
    rest = Dl - rad
    ui = Din - np.sum(Din * u0, -1, keepdims=True) * u0                 # input shift with its radial part removed
    ui = ui / (np.linalg.norm(ui, axis=-1, keepdims=True) + 1e-12)
    along = np.sum(rest * ui, -1, keepdims=True) * ui
    orth = rest - along
    nd, ni = np.linalg.norm(Dl, axis=-1), np.linalg.norm(Din, axis=-1); mv = (nd > 1e-6) & (ni > 1e-6)
    return {"d_nmse": float(np.sum((Pc - z) ** 2 - (P0 - z) ** 2) / zz),
            "d_nmse_shift_term": float(np.sum(Dl ** 2) / zz), "d_nmse_cross_term": float(2 * np.sum(Dl * (P0 - z)) / zz),
            "rel_shift": float(np.sum(Dl ** 2) / zz), "input_rel_shift": float(np.sum(Din ** 2) / zz),
            "frac_radial": float(np.sum(rad ** 2) / max(np.sum(Dl ** 2), 1e-12)),
            "frac_along_input": float(np.sum(along ** 2) / max(np.sum(Dl ** 2), 1e-12)),
            "frac_orth": float(np.sum(orth ** 2) / max(np.sum(Dl ** 2), 1e-12)),
            "radial_mean": float(np.mean(np.sum(Dl * u0, -1))),         # < 0 = predictions shrink
            "gain_real": float(np.sqrt(np.sum(Dl ** 2) / max(np.sum(Din ** 2), 1e-12))),
            "cos_follow": float(np.mean(np.sum(Dl * Din, -1)[mv] / (nd * ni)[mv])) if mv.any() else None,
            "n_moved_rows": int(mv.sum()), "shrink": float(np.mean(np.linalg.norm(Pc, axis=-1)) / np.mean(np.linalg.norm(P0, axis=-1)))}


def gain(m, Z, T, tte, pool, eps):               # paired directions: one fixed draw per K, shared by every model and pool
    n, K, d = Z.shape; rg = np.random.default_rng((5, K))
    Uc = rg.standard_normal((n, 1, d)).astype(np.float32); Ui = rg.standard_normal((n, K, d)).astype(np.float32)
    P0 = predict(m, Z, T, tte, pool); out = {}
    for tag, dZ in (("coherent", np.broadcast_to(Uc, Z.shape)), ("indep", Ui), ("one", Ui * (np.arange(K) == 0)[None, :, None])):
        dZ = (eps * dZ).astype(np.float32); P1 = predict(m, Z + dZ, T, tte, pool)
        ok = finite_rows(P1) & finite_rows(P0)
        mean_dz = dZ.mean(1)                     # per unit of the MEAN perturbation, so pooling alone does not set the scale
        out[f"gain_{tag}"] = float(np.sqrt(np.mean(np.sum((P1 - P0)[ok] ** 2, -1)) / max(np.mean(np.sum(mean_dz[ok] ** 2, -1)), 1e-12)))
    return out


def influence(m, Z, T, tte, occ_frac, disp):    # K views, mean pool; leave-one-view-out
    K = Z.shape[1]; C = contributions(m, Z, T, tte); P = pool_c(m, C, tte)
    infl = np.stack([np.mean(np.linalg.norm(P - pool_c(m, np.delete(C, j, 1), tte), axis=-1), -1) for j in range(K)], 1)
    cf = C.reshape(*C.shape[:2], -1); dev = np.linalg.norm(cf - cf.mean(1, keepdims=True), axis=-1)
    eff = infl / (dev + 1e-6)                    # influence per unit deviation from the pool (an effective weight)
    out = {"K": K}
    for cov_name, cov in (("occ", occ_frac), ("disp", disp)):
        use = (np.ptp(cov, 1) > 0) & (np.ptp(infl, 1) > 0)
        for q_name, q in (("infl", infl), ("effw", eff)):
            r = np.array([spearmanr(a, b)[0] for a, b in zip(q[use], cov[use])])
            out[f"rho_{q_name}_{cov_name}"] = float(np.nanmean(r)) if len(r) else None
        out[f"n_used_{cov_name}"] = int(use.sum())
    use = np.ptp(disp, 1) > 0; top = np.argmax(disp, 1)
    out["share_most_displaced"] = float(np.mean((infl[np.arange(len(infl)), top] / (infl.sum(1) + 1e-12))[use]))
    out["uniform_share"] = 1.0 / K
    return out


def contrib_centred(Cc, C0, cbar, disp):         # does a more-displaced view's contribution move toward the mean contribution?
    f = lambda C: np.linalg.norm((C - cbar).reshape(*C.shape[:2], -1), axis=-1)
    r = f(Cc) / (f(C0) + 1e-6)
    use = np.ptp(disp, 1) > 0
    rr = np.array([spearmanr(a, b)[0] for a, b in zip(r[use], disp[use])])
    q = np.quantile(disp, 0.9)
    return {"rho_centred_ratio_disp": float(np.nanmean(rr)) if len(rr) else None,
            "centred_ratio_top10pct_disp": float(r[disp >= q].mean()), "centred_ratio_rest": float(r[disp < q].mean())}


def build_models(seeds):
    models = S.load_models(seeds, (10,))
    # BP inference on G's weights uses the BP reference's selected step scale: at G's own c = 1 it oscillates (energy rises
    # on 37% of steps, output changes 32% with batch size; dbg job 669e6547), at the BP c it decreases monotonically.
    c_bp = sorted({S2.attach_bpg(m, n).cfg.c for n, m in models.items() if m.kind == "BPg"})
    assert len(c_bp) == 1, f"BP references disagree on the step scale: {c_bp}"
    for name in list(models):
        m = models[name]
        if m.kind == "BPg":
            models[name.replace("BPg_", "BPgMP_")] = MapPool(m.p, m.cfg, "MapPool", m.seed)      # BPg weights, mean of per-view MAP
        if m.kind == "G":
            gm = S2.GMap(m); gm.cfg = dc.replace(gm.cfg, c=c_bp[0]); gm.predict = G.make_predict(gm.p, None, gm.cfg)
            models[name.replace("G_", "Gmap_")] = gm                                                # G weights, joint MAP
            models[name.replace("G_", "GMP_")] = MapPool(gm.p, gm.cfg, "MapPool", m.seed)         # G weights, mean of per-view MAP
    for name, m in models.items():
        m.name = name
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--backbone", default=S.BACKBONE)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--n-infl", type=int, default=500); ap.add_argument("--kmax", type=int, default=48)
    a = ap.parse_args()
    ks = [k for k in KS if k <= a.kmax]
    L_ = np.load("pred_latents.npz")
    raw_tr = L_["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32)
    cfg, params = load_run(a.backbone); enc = S.Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32); HH = Xte.shape[1]
    assert len(Xte) == len(zte)
    views = {}
    for cond in ("clean", "noise", "occlude"):   # full test set first (views depend on chunking), then the subset
        Xc, occ = S.corrupt_images(Xte, cond)
        Z, T, B = S.whole_views(Xc, enc, cfg, max(ks), False, 303)
        of = (np.stack([S.occluded_frac(B[:, v, 1], B[:, v, 0], B[:, v, 2], occ, Hh=HH) for v in range(Z.shape[1])], 1)
              if occ is not None else np.zeros(Z.shape[:2]))
        views[cond] = (Z, T, B, of)
    for cond in ("noise", "occlude"):
        assert np.array_equal(views["clean"][1], views[cond][1]) and np.array_equal(views["clean"][2], views[cond][2]), "views not paired"
    if a.n:
        views = {k: tuple(x[:a.n] for x in v) for k, v in views.items()}; zte, tte = zte[:a.n], tte[:a.n]
    disp = {c: np.linalg.norm(views[c][0] - views["clean"][0], axis=-1) for c in ("noise", "occlude")}   # actual per-view latent displacement
    models = build_models(a.seeds)
    res = {"n": len(zte), "ks": ks, "dataset": cfg.data.dataset, "backbone": a.backbone, "complete": False,
           "seeds": {k: sorted({m.seed for m in models.values() if m.kind == k}) for k in {m.kind for m in models.values()}},
           "rows": [], "infl": [], "gain": [], "errors": [],
           "disp_occlude_mean": float(disp["occlude"].mean()), "disp_noise_mean": float(disp["noise"].mean()),
           "disp_occlude_q": np.quantile(disp["occlude"], [0.1, 0.5, 0.9]).tolist()}
    npz = a.out.replace(".json", "_pred.npz"); save = {"zte": zte.astype(np.float32)}
    for name, m in sorted(models.items()):
        t0 = time.time()
        try:
            pools = {"G": ["mean", "pw", "oracle"], "F": ["mean", "pw", "oracle"], "MapPool": ["mean", "oracle"]}.get(m.kind, ["own"])
            for K in ks:
                Z0, T0 = views["clean"][0][:, :K], views["clean"][1][:, :K]
                C0 = contributions(m, Z0, T0, tte)
                CC = {c: contributions(m, views[c][0][:, :K], T0, tte) for c in ("noise", "occlude")}   # computed once per K
                for pool in pools:
                    if pool in ("pw", "oracle") and K == 1:
                        continue
                    P0 = predict(m, Z0, T0, tte, "mean" if pool == "oracle" else pool, C0 if pool != "pw" else None)
                    base = {"model": name, "kind": m.kind, "seed": m.seed, "K": K, "pool": pool}
                    res["rows"].append({**base, "cond": "clean", "nmse": S.metrics(P0, zte)[0]["nmse"] if finite_rows(P0).all() else None,
                                        **geometry(P0, zte), "eff_rank": eff_rank(P0[finite_rows(P0)]), "cka_target": cka(P0[finite_rows(P0)], zte[finite_rows(P0)])})
                    if K == max(ks):
                        save[f"{name}|{pool}|clean"] = P0.astype(np.float32)
                    for cond in ("noise", "occlude"):
                        Zc, of = views[cond][0][:, :K], views[cond][3][:, :K]
                        Cc = CC[cond]
                        if pool == "oracle":             # down-weight by the known occluded fraction (noise: plain mean)
                            Pc = pool_c(m, Cc, tte, (1.0 - of).astype(np.float32) + 1e-3)
                        else:
                            Pc = predict(m, Zc, T0, tte, pool, Cc if pool != "pw" else None)
                        row = {**base, "cond": cond, "nmse": S.metrics(Pc, zte)[0]["nmse"] if finite_rows(Pc).all() else None,
                               **geometry(Pc, zte), **shift(Pc, P0, Zc, Z0, zte), "eff_rank": eff_rank(Pc[finite_rows(Pc)]),
                               "cka_target": cka(Pc[finite_rows(Pc)], zte[finite_rows(Pc)])}
                        if cond == "occlude" and K > 1 and pool == "mean" and C0 is not None:
                            row.update(contrib_centred(Cc, C0, C0.reshape(-1, *C0.shape[2:]).mean(0), disp[cond][:, :K]))
                        res["rows"].append(row)
                        if K == max(ks):
                            save[f"{name}|{pool}|{cond}"] = Pc.astype(np.float32)
                    if pool in ("mean", "own", "pw") and m.kind in ("G", "F", "BPg", "BPf"):    # C: gain, first N_GAIN images
                        ng = min(N_GAIN, len(zte)); sl = slice(0, ng)
                        g = gain(m, Z0[sl], T0[sl], tte[sl], pool, 0.1)
                        if K == 6:
                            g.update({k + "_eps0.05": v for k, v in gain(m, Z0[sl], T0[sl], tte[sl], pool, 0.05).items()})
                        res["gain"].append({**base, **g})
                S2.dump(res, a.out)
            if m.kind in ("G", "F", "BPf", "MapPool"):     # D: influence at K = 6 and 48, mean pool, first n_infl images
                for K in [k for k in (6, 48) if k in ks]:
                    ni = min(a.n_infl, len(zte)); Zc, Tc, _, of = views["occlude"]
                    res["infl"].append({"model": name, "kind": m.kind, "seed": m.seed,
                                        **influence(m, Zc[:ni, :K], Tc[:ni, :K], tte[:ni], of[:ni, :K], disp["occlude"][:ni, :K])})
        except Exception:
            res["errors"].append({"model": name, "traceback": traceback.format_exc()[-3000:]}); print(traceback.format_exc(), flush=True)
        S2.dump(res, a.out); np.savez_compressed(npz, **save)
        print(name, "done", round(time.time() - t0), "s", flush=True)
    res["complete"] = not res["errors"]
    S2.dump(res, a.out)
    print("DONE" if res["complete"] else f"DONE WITH {len(res['errors'])} ERRORS")
    sys.exit(0 if res["complete"] else 3)


if __name__ == "__main__":
    main()
