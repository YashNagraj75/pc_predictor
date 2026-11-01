import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax
jax.config.update("jax_default_matmul_precision", "highest")
import diag_saccade as S, diag_saccade2 as S2
from diag_saccade import D, load_run
from scipy.stats import spearmanr

# Latent analysis (Phase C, inference only): why do the PC predictors degrade differently from frozen BP under
# corruption?  Run from a dataset dir (Q1-PhaseA for Galaxy10, ds/<name> otherwise).  Uses diag_saccade2's views
# (key 303: clean and corrupted views share crops, so every comparison is paired) and its model loaders.
#  A  error geometry: nmse, prediction norm ratio ||P||/||z||, cosine(P, z)
#  B  corruption shift D = P_corrupt - P_clean: size, alignment with the input shift (mean seen latent), shrinkage
#  C  gain: rms change of P per rms perturbation of the seen latents (all views, or one view), clean views
#  D  per-view influence (leave-one-view-out) and contribution norm vs each view's occluded fraction
#  E  geometry: effective rank, linear CKA with targets; predictions saved for cross-model CKA
KS = (1, 6, 48)


def eff_rank(P):
    X = P.reshape(-1, P.shape[-1]); X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2; p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def cka(A, B):                                   # linear CKA over all (image, target) rows
    A = A.reshape(-1, A.shape[-1]).astype(np.float64); B = B.reshape(-1, B.shape[-1]).astype(np.float64)
    A, B = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def contributions(m, Z, T, tte):                 # per-view contribution before pooling: G embedding (n,K,M); F/BPf (n,K,4,d)
    if m.kind == "G":
        return m.embed(Z, T)
    if m.kind == "F":
        return S2.ff(m, Z, T, tte)
    return np.stack([S2._bp_predict(m, Z[:, v:v + 1], T[:, v:v + 1], tte, 1) for v in range(Z.shape[1])], 1)


def predict(m, Z, T, tte, pool="mean", C=None):
    K = Z.shape[1]
    if m.kind == "BPg":
        return S2._bp_predict(m, Z, T, tte, K)
    if pool == "pw" and K > 1:
        lp = S2.LPool(m, tte)
        for v in range(K):                       # one view at a time, own group, exactly as diag_saccade2.whole_pools
            lp.add(Z[:, v:v + 1], T[:, v:v + 1], -2 - v)
        return lp.predict("pw")[0]
    C = contributions(m, Z, T, tte) if C is None else C
    if m.kind == "G":
        return m.readout(C.mean(1), tte)
    return C.mean(1)                             # F and BPf: mean of per-view predictions


def geometry(P, z):
    nP, nz = np.linalg.norm(P, axis=-1), np.linalg.norm(z, axis=-1)
    return {"norm_ratio": float(nP.mean() / nz.mean()), "cos": float(np.mean(np.sum(P * z, -1) / (nP * nz + 1e-12)))}


def shift(Pc, P0, Zc, Z0, z):
    Dl = Pc - P0                                 # (n, 4, d)
    Din = (Zc.mean(1) - Z0.mean(1))[:, None]     # (n, 1, d) input shift, same latent space
    nd, ni = np.linalg.norm(Dl, axis=-1), np.linalg.norm(Din, axis=-1)
    return {"rel_shift": float(np.mean(nd ** 2) / np.mean(np.sum(z ** 2, -1))),
            "follow": float(np.mean(np.sum(Dl * Din, -1)) / (np.mean(ni ** 2) + 1e-12)),
            "cos_follow": float(np.mean(np.sum(Dl * Din, -1) / (nd * ni + 1e-12))),
            "input_rel_shift": float(np.mean(ni ** 2) / np.mean(np.sum(z ** 2, -1))),
            "shrink": float(np.mean(np.linalg.norm(Pc, axis=-1)) / np.mean(np.linalg.norm(P0, axis=-1)))}


def gain(m, Z, T, tte, pool, P0, eps, rng):
    U = rng.standard_normal(Z.shape).astype(np.float32)
    out = {}
    for tag, mask in (("all", np.ones(Z.shape[1])), ("one", np.eye(Z.shape[1])[0])):
        dZ = eps * U * mask[None, :, None]
        P1 = predict(m, Z + dZ, T, tte, pool)
        out[f"gain_{tag}"] = float(np.sqrt(np.mean(np.sum((P1 - P0) ** 2, -1)) / np.mean(np.sum(dZ[:, mask > 0] ** 2, -1))))
    return out


def influence(m, Z, T, tte, pool, occ_frac, C=None):  # leave-one-view-out influence vs occluded fraction, K views
    K = Z.shape[1]; P = predict(m, Z, T, tte, pool, C)
    infl = np.stack([np.mean(np.linalg.norm(P - predict(m, np.delete(Z, j, 1), np.delete(T, j, 1), tte, pool,
                     None if C is None else np.delete(C, j, 1)), axis=-1), -1) for j in range(K)], 1)   # (n, K)
    rho = [spearmanr(i, o)[0] for i, o in zip(infl, occ_frac) if np.ptp(o) > 0]
    top = np.argmax(occ_frac, 1)
    share = infl[np.arange(len(infl)), top] / (infl.sum(1) + 1e-12)
    return {"spearman_infl_occ": float(np.nanmean(rho)), "n_images_rho": len(rho),
            "share_most_occluded": float(np.mean(share)), "uniform_share": 1.0 / K}


def contrib_norm(m, Cc, C0, occ_frac):           # does a view's contribution shrink as it gets more occluded?
    if Cc is None:
        return {}
    r = np.linalg.norm(Cc.reshape(*Cc.shape[:2], -1), axis=-1) / (np.linalg.norm(C0.reshape(*C0.shape[:2], -1), axis=-1) + 1e-12)
    rho = [spearmanr(a, o)[0] for a, o in zip(r, occ_frac) if np.ptp(o) > 0]
    hi, lo = occ_frac > 0.5, occ_frac < 0.05
    return {"spearman_contrib_ratio_occ": float(np.nanmean(rho)),
            "contrib_ratio_occ_gt_0.5": float(r[hi].mean()) if hi.any() else None,
            "contrib_ratio_occ_lt_0.05": float(r[lo].mean()) if lo.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--backbone", default=S.BACKBONE)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--n-infl", type=int, default=500); ap.add_argument("--k-infl", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=48)
    a = ap.parse_args()
    ks = [k for k in KS if k <= a.kmax]
    L_ = np.load("pred_latents.npz")
    raw_tr = L_["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32)
    cfg, params = load_run(a.backbone); enc = S.Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32); HH = Xte.shape[1]
    assert len(Xte) == len(zte)
    if a.n:
        Xte, zte, tte = Xte[:a.n], zte[:a.n], tte[:a.n]
    models = S.load_models(a.seeds, (10,))
    for name, m in models.items():
        if m.kind == "BPg":
            S2.attach_bpg(m, name)
    views = {}
    for cond in ("clean", "noise", "occlude"):
        Xc, occ = S.corrupt_images(Xte, cond)
        Z, T, B = S.whole_views(Xc, enc, cfg, max(ks), False, 303)
        of = np.stack([S.occluded_frac(B[:, v, 1], B[:, v, 0], B[:, v, 2], occ, Hh=HH) for v in range(Z.shape[1])], 1) if occ is not None else None
        views[cond] = (Z, T, of)
    assert np.array_equal(views["clean"][1], views["occlude"][1]), "clean and corrupted views must share crops (paired)"
    res = {"n": len(zte), "ks": ks, "dataset": cfg.data.dataset, "backbone": a.backbone, "rows": [], "infl": [], "geom": []}
    save = {}
    rng = np.random.default_rng(5)
    for name, m in sorted(models.items()):
        t0 = time.time()
        pools = ["mean", "pw"] if m.kind in ("G", "F") else ["own"]
        for K in ks:
            Z0, T0, _ = views["clean"]; Z0, T0 = Z0[:, :K], T0[:, :K]
            C0 = contributions(m, Z0, T0, tte) if m.kind != "BPg" else None
            for pool in pools:
                if pool == "pw" and K == 1:
                    continue
                P0 = predict(m, Z0, T0, tte, pool, C0 if pool != "pw" else None)
                base = {"model": name, "kind": m.kind, "K": K, "pool": pool}
                res["rows"].append({**base, "cond": "clean", "nmse": S.metrics(P0, zte)[0]["nmse"], **geometry(P0, zte),
                                    **gain(m, Z0, T0, tte, pool, P0, 0.1, rng), **{k + "_eps0.05": v for k, v in gain(m, Z0, T0, tte, pool, P0, 0.05, rng).items()}})
                res["geom"].append({**base, "cond": "clean", "eff_rank": eff_rank(P0), "cka_target": cka(P0, zte)})
                if K == max(ks):
                    save[f"{name}|{pool}|clean"] = P0.astype(np.float16)
                for cond in ("noise", "occlude"):
                    Zc, Tc, of = views[cond]; Zc = Zc[:, :K]
                    Cc = contributions(m, Zc, T0, tte) if m.kind != "BPg" else None
                    Pc = predict(m, Zc, T0, tte, pool, Cc if pool != "pw" else None)
                    row = {**base, "cond": cond, "nmse": S.metrics(Pc, zte)[0]["nmse"], **geometry(Pc, zte), **shift(Pc, P0, Zc, Z0, zte)}
                    if cond == "occlude" and K > 1 and pool in ("mean", "own"):
                        row.update(contrib_norm(m, Cc, C0, of[:, :K]))
                    res["rows"].append(row)
                    res["geom"].append({**base, "cond": cond, "eff_rank": eff_rank(Pc), "cka_target": cka(Pc, zte)})
                    if K == max(ks):
                        save[f"{name}|{pool}|{cond}"] = Pc.astype(np.float16)
            S2.dump(res, a.out)
        # D: leave-one-view-out influence under occlusion, K = k_infl, first n_infl images
        Zc, Tc, of = views["occlude"]; ni, ki = min(a.n_infl, len(zte)), a.k_infl
        for pool in pools:
            Zs, Ts, ts = Zc[:ni, :ki], Tc[:ni, :ki], tte[:ni]
            Cs = contributions(m, Zs, Ts, ts) if (m.kind != "BPg" and pool != "pw") else None
            res["infl"].append({"model": name, "kind": m.kind, "pool": pool, "K": ki, **influence(m, Zs, Ts, ts, pool, of[:ni, :ki], Cs)})
        S2.dump(res, a.out)
        print(name, "done", round(time.time() - t0), "s", flush=True)
    np.savez_compressed(a.out.replace(".json", "_pred.npz"), zte=zte.astype(np.float16), **save)
    print("DONE")


if __name__ == "__main__":
    main()
