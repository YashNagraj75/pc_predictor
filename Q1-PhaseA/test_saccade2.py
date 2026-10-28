import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import metrics as MT
import diag_saccade as S, diag_saccade2 as S2
from diag_saccade import D, load_run

# Gates for diag_saccade2.py (rev 2, after 3 reviews); test_saccade.py runs first in the job.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def rel(a, b):                                   # element-wise, floored at 1e-3 x median |b| (review NUM N8)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-3 * np.median(np.abs(b)) + 1e-12)))


def brute_loo(m, pool, groups):                  # leave-out error of every latent, leaving out its whole group
    N = len(pool.w); out = np.zeros((pool.n, N))
    for i in range(N):
        keep = np.array([pool.g[j] != pool.g[i] if groups else j != i for j in range(N)]); w = pool.w[keep]
        if m.kind == "G":
            zh = m.readout(np.einsum("j,njm->nm", w, pool.C[:, keep]) / w.sum(), pool.T[:, i:i + 1])[:, 0]
        else:
            zh = np.einsum("j,njbd->nbd", w, S2.ff(m, pool.Z[:, keep], pool.T[:, keep], pool.T[:, i:i + 1]))[:, 0] / w.sum()
        out[:, i] = np.sum((zh - pool.Z[:, i]) ** 2, -1)
    return out


def pooled(m, pool, p):                          # pooled prediction with explicit weights p (n, N)
    c = np.einsum("nj,nj...->n...", p, pool.C) / p.sum(1).reshape((-1,) + (1,) * (pool.C.ndim - 2))
    return m.readout(c, pool.tte) if m.kind == "G" else c


def build(m, Zc, Tc, t_, fix, geom, shift=None):
    n = len(t_); pool = S2.LPool(m, t_)
    pool.add(Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0), -1); pool.set_peri(0)
    dpre = []
    for k in range(fix.shape[1]):
        ids = geom["crops"][fix[:, k]]; z = np.take_along_axis(Zc, ids[..., None], 1).copy(); t = Tc[ids]
        if shift is not None and k == 1:         # synthetic corrupted glimpse: 5 mutually consistent, off-image latents
            z = z.mean(1, keepdims=True) + 0 * z + shift
        ph, o2n = pool.pred_at(t); dpre.append(np.mean(np.sum((ph - z) ** 2, -1), 1))
        pool.add(z, t, k, None, o2n)
        if k >= 1:
            pool.bump_peri()
    return pool, np.stack(dpre, 1)


def main():
    L_ = np.load("pred_latents.npz")
    raw_tr = L_["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte, yte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32), np.asarray(L_["yte"])
    ztr_n, ytr = (raw_tr - mu) / sd, np.asarray(L_["ytr"])
    knn = S2.KNN(ztr_n, ytr)
    # 1. kNN == metrics.knn_accuracy on 4000 noisy float64 queries (crosses chunk boundaries)
    Q = zte[:1000] + 0.5 * np.random.default_rng(0).standard_normal(zte[:1000].shape)
    a = knn(Q, yte[:1000]); b = MT.knn_accuracy(ztr_n.reshape(-1, 128), np.repeat(ytr, 4), Q.reshape(-1, 128), np.repeat(yte[:1000], 4), k=20)
    check("kNN == metrics.knn_accuracy (4000 noisy queries)", abs(a - b) < 0.002, f"{a:.4f} vs {b:.4f}")
    cfg, params = load_run(S.BACKBONE); enc = S.Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); n = 24
    X = np.asarray(Xte[:n], np.float32); z_, t_, y_ = zte[:n], tte[:n], yte[:n]
    Zc, Tc = S.crop_table(X, enc, cfg.data.img_size)
    geom = S.make_geom("grid4"); models = {"G": S.GPred(10, 0), "F": S.FPred(0)}
    fix = np.random.default_rng(3).integers(0, geom["L"], (n, 3))
    for kind, m in models.items():
        # 2. surprise and mean pool == findings/31 Pool
        old = S.Pool(m, Zc, Tc, t_); new = S2.LPool(m, t_)
        new.add(Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0), -1); new.set_peri(0)
        for k in range(3):
            ids = geom["crops"][fix[:, k]]; z = np.take_along_axis(Zc, ids[..., None], 1); t = Tc[ids]
            ph, o2n = new.pred_at(t); e_new = np.mean(np.sum((ph - z) ** 2, -1), 1); e_old = old.surprise(z, t)
            check(f"{kind} step {k}: surprise == findings/31 Pool", rel(e_new, e_old) < 1e-4, f"{rel(e_new, e_old):.1e}")
            old.add(z, t, geom["kinds"]); new.add(z, t, k, None, o2n)
            if k >= 1:
                new.bump_peri()
        P0 = new.predict("mean")[0]
        check(f"{kind}: mean pool == findings/31 Pool 'full'", rel(P0, old.predict("full")) < 1e-3, f"{rel(P0, old.predict('full')):.1e}")
        # 3. leave-one-latent-out and leave-one-glimpse-out errors == brute force (element-wise)
        bl, bg = brute_loo(m, new, False), brute_loo(m, new, True)
        check(f"{kind}: leave-one-latent-out == brute force", rel(new.loo(), bl) < 1e-3, f"{rel(new.loo(), bl):.1e}")
        check(f"{kind}: leave-one-glimpse-out == brute force", rel(new.loo_group(), bg) < 1e-3, f"{rel(new.loo_group(), bg):.1e}")
        for gid in (1, 2):
            idx = np.where(new.g == gid)[0]
            check(f"{kind}: glimpse_err({gid}) == mean brute force", rel(new.glimpse_err(gid), bg[:, idx].mean(1)) < 1e-3)
        # 4. precision pools == hand formula
        for md, d in (("pw", bl), ("pwg", bg)):
            p = new.w[None] / (d + S2.LAM * np.median(d, 1, keepdims=True) + 1e-12)
            check(f"{kind}: '{md}' pool == hand-computed precision pool", rel(new.predict(md)[0], pooled(m, new, p)) < 1e-3)
        # 5. ig weights == hand computation (latest glimpse gets the image's mean older reducibility)
        pool, dpre = build(m, Zc, Tc, t_, fix, geom)
        e = np.stack([pool.glimpse_err(j) for j in range(3)], 1); r = np.clip(1 - e / dpre, 0, 1); r[:, 2] = r[:, :2].mean(1)
        check(f"{kind}: ig_weights == hand computation", rel(S2.ig_weights(dpre, pool, 3), e * r) < 1e-6)
        # 6. a synthetic corrupted glimpse: group precision down-weights it more than latent precision does
        poolc, _ = build(m, Zc, Tc, t_, fix, geom, shift=3.0)
        idx = np.where(poolc.g == 1)[0]
        share = {md: float(np.mean((lambda w: w[:, idx].sum(1) / w.sum(1))(poolc.weights(md)))) for md in ("mean", "pw", "pwg")}
        # both precision pools must cut the corrupted glimpse to under half its mean-pool share; which is better is a result
        check(f"{kind}: corrupted glimpse share pw, pwg < 0.5 x mean", max(share["pw"], share["pwg"]) < 0.5 * share["mean"],
              str({k_: round(v, 3) for k_, v in share.items()}))
    # 7. run_saccades == findings/31 code (random, epi); every policy and pool finite incl. occlusion; ig vs epi
    for kind, m in models.items():
        for pol in ("random", "epi"):
            o_old, *_ = S.run_saccades(m, Zc, Tc, t_, z_, X, geom, pol, 4, None, np.random.default_rng(7), {})
            o_new, _, _ = S2.run_saccades(m, Zc, Tc, t_, z_, y_, X, geom, pol, 4, None, np.random.default_rng(7), knn, [2, 4])
            a, b = o_new[(4, "full")]["nmse"], o_old[(4, "full")]["nmse"]
            check(f"{kind} {pol}: 4-glimpse nmse == findings/31 code", abs(a - b) < 1e-5, f"{a:.6f} vs {b:.6f}")
        Xo, occ = S.corrupt_images(X, "occlude")
        Zo, To = S.crop_table(Xo, enc, cfg.data.img_size)
        for pol in S2.POLICIES + ["oracle_occ"]:
            o, st, hi = S2.run_saccades(m, Zo, To, t_, z_, y_, Xo, geom, pol, 6, occ, np.random.default_rng(7), knn, [6])
            vals = [r["nmse"] for (k, pl), r in o.items() if k != "diag"]
            check(f"{kind} occlude {pol}: all pools finite, no revisits", all(v is not None and np.isfinite(v) for v in vals) and st["revisits"] == 0)
            if pol == "epi":
                h_epi = hi
            if pol == "ig":
                print(f"     report: {kind} ig differs from epi at fixations 3-6 in {np.mean(np.any(hi[:, 2:] != h_epi[:, 2:], 1)):.2f} of images")
    # 8. whole-view pools: mean == score_whole; latent LOO (m = 1) == brute force; K=1 pw == mean
    Zv, Tv, Bv = S.whole_views(X, enc, cfg, 6, False, 303)
    for kind, m in models.items():
        o = S2.whole_pools(m, Zv, Tv, Bv, t_, z_, y_, knn, [1, 3], None)
        ref = S.score_whole(m, Zv, Tv, t_, z_, [3])[3][0]["nmse"]
        check(f"{kind}: whole-view mean pool == score_whole at k=3", abs(o[(3, "mean")]["nmse"] - ref) < 1e-5)
        check(f"{kind}: whole-view K=1 pw == mean", abs(o[(1, "pw")]["nmse"] - o[(1, "mean")]["nmse"]) < 1e-7)
        wp = S2.LPool(m, t_)
        for v in range(6):
            wp.add(Zv[:, v:v + 1], Tv[:, v:v + 1], -2 - v)
        check(f"{kind}: whole-view leave-one-out == brute force", rel(wp.loo(), brute_loo(m, wp, False)) < 1e-3)
    # 9. BP and BP-inference rows: k=1 reproduces, finite at k=6; convergence report T vs 4T
    g = models["G"]; gm = S2.GMap(g); bpg = S2.attach_bpg(S.BPg(10, 1), "BPg_d10_s1"); bpf = S.BPf(0)
    for name, m in (("BPg_d10_s1", bpg), ("BPf_d10_s0", bpf), ("Gmap_d10_s0", gm)):
        P1, P6 = S2._bp_predict(m, Zv, Tv, t_, 1), S2._bp_predict(m, Zv, Tv, t_, 6)
        ok = np.all(np.isfinite(P6))
        if name != "Gmap_d10_s0":
            ok &= abs(S.metrics(P1, z_)[0]["nmse"] - S.score_whole(m, Zv, Tv, t_, z_, [1])[1][0]["nmse"]) < 1e-5
        check(f"{name}: k=1 == score_whole, k=6 finite", ok)
        if name != "BPf_d10_s0":
            P4 = S2._bp_predict(m, Zv, Tv, t_, 6, T=4 * m.cfg.T)
            print(f"     report: {name} k=6 nmse at T {S.metrics(P6, z_)[0]['nmse']:.4f} vs 4T {S.metrics(P4, z_)[0]['nmse']:.4f}")
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
