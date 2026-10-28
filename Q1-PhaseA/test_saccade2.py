import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import metrics as MT
import diag_saccade as S, diag_saccade2 as S2
from diag_saccade import D, load_run

# Gates for diag_saccade2.py; all must pass before the full run (host GPU).  test_saccade.py runs first in the job.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    L_ = np.load("pred_latents.npz")
    raw_tr = L_["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte, yte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32), np.asarray(L_["yte"])
    ztr_n, ytr = (raw_tr - mu) / sd, np.asarray(L_["ytr"])
    knn = S2.KNN(ztr_n, ytr)
    # 1. GPU kNN == metrics.knn_accuracy (cosine, k = 20)
    a = knn(zte[:500], yte[:500])
    b = MT.knn_accuracy(ztr_n.reshape(-1, 128), np.repeat(ytr, 4), zte[:500].reshape(-1, 128), np.repeat(yte[:500], 4), k=20)
    check("kNN == metrics.knn_accuracy", abs(a - b) < 0.005, f"{a:.4f} vs {b:.4f}")
    cfg, params = load_run(S.BACKBONE); enc = S.Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); n = 24
    X = np.asarray(Xte[:n], np.float32); z_, t_, y_ = zte[:n], tte[:n], yte[:n]
    Zc, Tc = S.crop_table(X, enc, cfg.data.img_size)
    geom = S.make_geom("grid4"); models = {"G": S.GPred(10, 0), "F": S.FPred(0)}
    rng = np.random.default_rng(3); fix = rng.integers(0, geom["L"], (n, 3))
    for kind, m in models.items():
        old = S.Pool(m, Zc, Tc, t_)
        new = S2.LPool(m, t_); new.add(Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0), -1); new.set_peri(0)
        for k in range(3):
            ids = geom["crops"][fix[:, k]]; z = np.take_along_axis(Zc, ids[..., None], 1); t = Tc[ids]
            e_old = old.surprise(z, t); e_new = np.mean(np.sum((new.pred_at(t) - z) ** 2, -1), 1)
            check(f"{kind} step {k}: surprise == findings/31 Pool", rel(e_new, e_old) < 1e-4, f"{rel(e_new, e_old):.1e}")
            old.add(z, t, geom["kinds"]); new.add(z, t, k)
            if k >= 1:
                new.bump_peri()
        check(f"{kind}: mean pool after 3 glimpses == findings/31 Pool 'full'", rel(new.predict(False), old.predict("full")) < 1e-4,
              f"{rel(new.predict(False), old.predict('full')):.1e}")
        # 2. leave-one-out error of every latent == brute force
        N = len(new.w); bf = np.zeros((n, N))
        for i in range(N):
            keep = np.arange(N) != i; w = new.w[keep]
            if kind == "G":
                s = np.einsum("j,njm->nm", w, new.C[:, keep]) / w.sum()
                zh = m.readout(s, new.T[:, i:i + 1])[:, 0]
            else:
                zh = np.einsum("j,njbd->nbd", w, S2.ff(m, new.Z[:, keep], new.T[:, keep], new.T[:, i:i + 1]))[:, 0] / w.sum()
            bf[:, i] = np.sum((zh - new.Z[:, i]) ** 2, -1)
        check(f"{kind}: leave-one-out errors == brute force (all {N} latents)", rel(new.loo(), bf) < 1e-4, f"{rel(new.loo(), bf):.1e}")
        # 3. leave-one-glimpse-out error == brute force
        idx = np.where(new.g == 1)[0]; keep = np.setdiff1d(np.arange(N), idx); w = new.w[keep]
        if kind == "G":
            zh = m.readout(np.einsum("j,njm->nm", w, new.C[:, keep]) / w.sum(), new.T[:, idx])
        else:
            zh = np.einsum("j,njbd->nbd", w, S2.ff(m, new.Z[:, keep], new.T[:, keep], new.T[:, idx])) / w.sum()
        e_bf = np.mean(np.sum((zh - new.Z[:, idx]) ** 2, -1), 1)
        check(f"{kind}: glimpse-left-out error == brute force", rel(new.glimpse_err(1), e_bf) < 1e-4, f"{rel(new.glimpse_err(1), e_bf):.1e}")
        # 4. precision weighting: huge LAM -> uniform weights -> equals the mean pool; normal LAM differs
        lam = S2.LAM; S2.LAM = 1e9
        d_u = rel(new.predict(True), new.predict(False)); S2.LAM = lam
        check(f"{kind}: precision pool with uniform precision == mean pool", d_u < 1e-4, f"{d_u:.1e}")
        check(f"{kind}: precision pool differs from the mean pool at LAM {lam}", rel(new.predict(True), new.predict(False)) > 1e-4)
    # 5. run_saccades (new) == findings/31 run_saccades for random and epi (full pool nmse), and ig == epi for 2 steps
    Xo = X
    for kind, m in models.items():
        for pol in ("random", "epi"):
            o_old, *_ = S.run_saccades(m, Zc, Tc, t_, z_, Xo, geom, pol, 4, None, np.random.default_rng(7), {})
            o_new, _, hi = S2.run_saccades(m, Zc, Tc, t_, z_, y_, Xo, geom, pol, 4, None, np.random.default_rng(7), knn, [2, 4])
            a, b = o_new[(4, "full")]["nmse"], o_old[(4, "full")]["nmse"]
            check(f"{kind} {pol}: 4-glimpse nmse == findings/31 code", abs(a - b) < 1e-5, f"{a:.6f} vs {b:.6f}")
        _, _, h_epi = S2.run_saccades(m, Zc, Tc, t_, z_, y_, Xo, geom, "epi", 4, None, np.random.default_rng(7), knn, [4])
        _, _, h_ig = S2.run_saccades(m, Zc, Tc, t_, z_, y_, Xo, geom, "ig", 4, None, np.random.default_rng(7), knn, [4])
        check(f"{kind}: ig == epi for the first 2 fixations, no revisits", np.all(h_ig[:, :2] == h_epi[:, :2]) and
              all(len(set(r)) == len(r) for r in h_ig))
    # 6. PC whole-view pool (mean) == model.predict / per-view average at k = 3; BP multi-view runs and k=1 reproduces
    Zv, Tv, _ = S.whole_views(X, enc, cfg, 3, False, 303)
    for kind, m in models.items():
        o = S2.whole_pools(m, Zv, Tv, t_, z_, y_, knn, [3])
        ref = S.score_whole(m, Zv, Tv, t_, z_, [3])[3][0]["nmse"]
        check(f"{kind}: whole-view mean pool == score_whole at k=3", abs(o[(3, "mean")]["nmse"] - ref) < 1e-5, f"{o[(3, 'mean')]['nmse']:.6f} vs {ref:.6f}")
    for name, m in (("BPg_d10_s1", S.BPg(10, 1)), ("BPf_d10_s0", S.BPf(0))):
        P1 = S2._bp_predict(m, Zv, Tv, t_, 1); P3 = S2._bp_predict(m, Zv, Tv, t_, 3)
        ref = S.score_whole(m, Zv, Tv, t_, z_, [1])[1][0]["nmse"]
        check(f"{name}: k=1 == score_whole; k=3 finite", abs(S.metrics(P1, z_)[0]["nmse"] - ref) < 1e-5 and np.all(np.isfinite(P3)))
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
