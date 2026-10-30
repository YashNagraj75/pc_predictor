import argparse, dataclasses as dc, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
import diag_saccade as S, pred_gpc as G
from diag_saccade import D, load_run

# Follow-up to findings/31, inference only (user 2026-10-28), rev 2 after 3 reviews.  On the grid4 front end it adds
#  - kNN class probe of the predicted target latents (cosine, k = 20, vs training backbone latents) for every row;
#  - precision-weighted pools, weight w_i / (delta_i + LAM median delta): delta by leave-one-latent-out ('pw') and by
#    leave-one-GLIMPSE-out ('pwg', siblings removed), plus shuffled-precision and occlusion-oracle controls;
#  - planner 'ig' (each glimpse: unexplained error x fraction removed by later evidence) with ig_shuf / ior controls;
#  - frozen BP on 1..192 whole views (same weights), and the PC-G weights under BP's own inference ('map').
KS = [1, 2, 4, 8, 16, 32]
KV = [1, 3, 6, 12, 24, 48, 96, 192]
POLICIES = ["random", "epi", "epi_shuf", "ig", "ig_shuf", "ior", "bu"]
LAM = 0.1
HH = 69                                        # image side for occluded_frac; set from the data in main()
DIAG_SEED = 1                                  # Gmap rows and the T vs 4T check run on this seed only (rev 3, cost)
ROWS = 65536                                   # multiple of 8192 (make_predict_x's chunk): padded to avoid recompiles


class KNN:
    def __init__(self, ztr, ytr, k=20):
        assert int(ytr.max()) < 10
        a = ztr.reshape(-1, ztr.shape[-1]).astype(np.float32)
        self.a = jnp.asarray(a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12))
        self.y = jnp.asarray(np.repeat(ytr, ztr.shape[1])); self.k = k
        self._f = jax.jit(lambda b, a, y: jnp.argmax(jax.nn.one_hot(y[jax.lax.top_k(
            (b / (jnp.linalg.norm(b, axis=1, keepdims=True) + 1e-12)) @ a.T, k)[1]], 10).sum(1), 1))

    def predict(self, Q, ch=512):
        Q = np.asarray(Q, np.float32); n = len(Q); pad = (-n) % ch
        Q = np.concatenate([Q, np.zeros((pad, Q.shape[1]), np.float32)])
        return np.concatenate([np.asarray(self._f(jnp.asarray(Q[i:i + ch]), self.a, self.y)) for i in range(0, len(Q), ch)])[:n]

    def correct(self, P, yte):                  # (n, 4, d) -> (n, 4) bool, plus predicted labels
        lab = self.predict(P.reshape(-1, P.shape[-1])).reshape(P.shape[0], -1)
        return lab == yte[:, None], lab

    def __call__(self, P, yte):
        return float(np.mean(self.correct(P, yte)[0]))


def ff(model, z, ta, tb):                       # F on every (latent a, target tau b) pair: (n,a,d),(n,a,9),(n,b,9) -> (n,a,b,d)
    n, A, B = z.shape[0], z.shape[1], tb.shape[1]
    zr = np.repeat(z[:, :, None], B, 2).reshape(-1, z.shape[-1]); tar = np.repeat(ta[:, :, None], B, 2).reshape(-1, 9)
    tbr = np.repeat(tb[:, None], A, 1).reshape(-1, 9)
    out = np.empty((len(zr), z.shape[-1]), np.float32)
    for i in range(0, len(zr), ROWS):
        j = min(i + ROWS, len(zr)); pad = (-(j - i)) % 8192
        x = np.concatenate([zr[i:j], tar[i:j], tbr[i:j]], -1)
        x = np.concatenate([x, np.zeros((pad, x.shape[1]), x.dtype)]) if pad else x
        out[i:j] = model.px(x)[:j - i]
    return out.reshape(n, A, B, -1)


# A set of seen latents with multiplicities w_i (peripheral counted once per glimpse, as in MPC), group ids g_i
# (glimpse index; whole views and the peripheral latent are their own groups), and occluded fractions o_i.
# Per latent: contribution c_i (G: amortiser embedding; F: predictions of the 4 targets); for F also
# R_i = sum_{j != i} w_j f(z_j, tau_j, tau_i) and I_i (same-group part), so leave-out errors cost O(new x all).
class LPool:
    def __init__(self, model, tte):
        self.m, self.tte, self.n = model, tte, len(tte)
        self.Z = self.T = self.C = self.R = self.I = self.Fp = None
        self.w = np.zeros(0); self.g = np.zeros(0, int); self.o = np.zeros((self.n, 0)); self.peri = None

    def pred_at(self, t):                       # current pooled prediction of latents at taus t; F also returns pairs
        if self.m.kind == "G":
            return self.m.readout(self.mean_c(), t), None
        o2n = ff(self.m, self.Z, self.T, t)
        return np.einsum("j,njbd->nbd", self.w, o2n) / self.w.sum(), o2n

    def mean_c(self, wts=None):
        wts = np.broadcast_to(self.w, (self.n, len(self.w))) if wts is None else wts
        return np.einsum("nj,nj...->n...", wts, self.C) / wts.sum(1).reshape((-1,) + (1,) * (self.C.ndim - 2))

    def add(self, z, t, gid, occ=None, o2n=None):
        m = z.shape[1]
        c = self.m.embed(z, t) if self.m.kind == "G" else ff(self.m, z, t, self.tte)
        if self.m.kind == "F":
            d = z.shape[-1]
            if self.Z is None:
                Rn = np.zeros((self.n, m, d)); Fp_new = np.zeros((self.n, m, d))
            else:
                o2n = ff(self.m, self.Z, self.T, t) if o2n is None else o2n            # (n, N, m, d)
                Rn = np.einsum("j,njbd->nbd", self.w, o2n)
                self.R = self.R + ff(self.m, z, t, self.T).sum(1)
                Fp_new = o2n[:, self.peri] if self.peri is not None else np.zeros_like(Rn)
            In = (ff(self.m, z, t, t) * (1 - np.eye(m))[None, :, :, None]).sum(1) if m > 1 else np.zeros((self.n, m, d))
            Rn = Rn + In
            self.R = Rn if self.R is None else np.concatenate([self.R, Rn], 1)
            self.I = In if self.I is None else np.concatenate([self.I, In], 1)
            self.Fp = Fp_new if self.Fp is None else np.concatenate([self.Fp, Fp_new], 1)
        cat = lambda a, b: b if a is None else np.concatenate([a, b], 1)
        self.Z, self.T, self.C = cat(self.Z, z), cat(self.T, t), cat(self.C, c)
        self.w = np.concatenate([self.w, np.ones(m)]); self.g = np.concatenate([self.g, np.full(m, gid)])
        self.o = np.concatenate([self.o, np.zeros((self.n, m)) if occ is None else occ], 1)

    def set_peri(self, idx):
        self.peri = idx
        if self.m.kind == "F":
            self.Fp = ff(self.m, self.Z[:, idx:idx + 1], self.T[:, idx:idx + 1], self.T)[:, 0]; self.Fp[:, idx] = 0.0

    def bump_peri(self):
        self.w[self.peri] += 1
        if self.m.kind == "F":
            self.R = self.R + self.Fp

    def _err(self, zh, idx):
        return np.sum((zh - self.Z[:, idx]) ** 2, -1)

    def loo(self):                              # leave-one-latent-out squared error, (n, N)
        W = self.w.sum()
        if self.m.kind == "G":
            S_ = np.einsum("j,njm->nm", self.w, self.C)
            s = (S_[:, None] - self.w[None, :, None] * self.C) / (W - self.w)[None, :, None]
            zh = self.m.readout(s.reshape(-1, s.shape[-1]), self.T.reshape(-1, 1, 9))[:, 0].reshape(self.n, len(self.w), -1)
        else:
            zh = self.R / (W - self.w)[None, :, None]
        return self._err(zh, slice(None))

    def loo_group(self):                        # leave-one-GROUP-out error of every latent, (n, N)
        W = self.w.sum(); out = np.zeros((self.n, len(self.w)))
        if self.m.kind == "G":
            S_ = np.einsum("j,njm->nm", self.w, self.C)
        for gid in np.unique(self.g):
            idx = np.where(self.g == gid)[0]; Wg = self.w[idx].sum()
            if self.m.kind == "G":
                s = (S_ - np.einsum("j,njm->nm", self.w[idx], self.C[:, idx])) / (W - Wg)
                zh = self.m.readout(s, self.T[:, idx])
            else:
                zh = (self.R[:, idx] - self.I[:, idx]) / (W - Wg)
            out[:, idx] = self._err(zh, idx)
        return out

    def glimpse_err(self, gid):                 # mean leave-glimpse-out error of glimpse gid, (n,)
        idx = np.where(self.g == gid)[0]; W = self.w.sum() - self.w[idx].sum()
        if self.m.kind == "G":
            S_ = np.einsum("j,njm->nm", self.w, self.C) - np.einsum("j,njm->nm", self.w[idx], self.C[:, idx])
            zh = self.m.readout(S_ / W, self.T[:, idx])
        else:
            zh = (self.R[:, idx] - self.I[:, idx]) / W
        return np.mean(self._err(zh, idx), 1)

    def weights(self, mode, rng=None):          # per-image pooling weights (n, N), multiplicities included
        if mode == "mean" or len(self.w) < 2:
            return np.broadcast_to(self.w, (self.n, len(self.w))).copy()
        if mode == "oracle":
            return self.w[None] * np.clip(1.0 - self.o, 0.0, 1.0) + 1e-6
        d = self.loo() if mode == "pw" else self.loo_group()
        p = 1.0 / (d + LAM * np.median(d, 1, keepdims=True) + 1e-12)
        if mode == "pwg_shuf":                  # same weight distribution, assignment to latents destroyed
            p = rng.permuted(p, axis=1)
        return self.w[None] * p

    def predict(self, mode="mean", rng=None):
        wts = self.weights(mode, rng); c = self.mean_c(wts)
        return (self.m.readout(c, self.tte) if self.m.kind == "G" else c), wts


# ig: glimpse j gets (unexplained error e_j, whole glimpse left out) x (fraction of its arrival error removed by
# later evidence, r_j).  The newest glimpse has no later evidence: it gets the image's mean r over older glimpses,
# so all glimpses are on one scale (review DESIGN F3).
def ig_weights(dpre, pool, k):
    e = np.stack([pool.glimpse_err(j) for j in range(k)], 1)
    r = np.clip(1.0 - e / (dpre[:, :k] + 1e-12), 0.0, 1.0)
    r[:, k - 1] = r[:, :k - 1].mean(1)
    return e * r


def finite(P):
    return bool(np.all(np.isfinite(P)))


def score(P, zte, yte, knn, per=None, key=None):
    if not finite(P):
        return {"nmse": None, "knn": None, "n_nonfinite": int(np.sum(~np.isfinite(P)))}
    m, sq = S.metrics(P, zte); ok, lab = knn.correct(P, yte)
    if per is not None:
        per[key + "|sq"] = sq; per[key + "|knn"] = ok.mean(1).astype(np.float32)
    return {"nmse": m["nmse"], "knn": float(ok.mean()), "knn_img_major": float(np.mean(
        [np.bincount(r, minlength=10).argmax() == y for r, y in zip(lab, yte)])),
        "pred_class_hist": np.bincount(lab.ravel(), minlength=10).tolist()}


def wdiag(wts, pool, occ_on):                   # weight diagnostics: ESS fraction, peripheral share, occluded mass
    w = wts / wts.sum(1, keepdims=True)
    out = {"ess_frac": float(np.mean(1.0 / np.sum(w ** 2, 1) / w.shape[1]))}
    if pool.peri is not None:
        out["peri_share"] = float(w[:, pool.peri].mean())
    if occ_on:
        out["occluded_mass"] = float(np.mean(np.sum(w * (pool.o > 0.5), 1)))
        out["occluded_count_frac"] = float(np.mean(np.sum((pool.o > 0.5) * pool.w[None], 1) / pool.w.sum()))
    return out


def run_saccades(model, Zc, Tc, tte, zte, yte, X, geom, policy, Kmax, occ, rng, knn, ks=None, per=None, key=""):
    ks = ks or KS
    n, L, cr = len(zte), geom["L"], geom["crops"]
    bu = S.bottom_up(X, geom); sp = S.crop_specs()
    perm = rng.permuted(np.tile(np.arange(L), (n, 1)), axis=1)
    occ_crop = (np.stack([S.occluded_frac(sp[c][0], sp[c][1], sp[c][2], occ) for c in range(S.NCROP)], 1)
                if occ is not None else None)                                   # (n, NCROP)
    occ_score = np.stack([occ_crop[:, cr[l]].mean(1) for l in range(L)], 1) if occ is not None else None
    pool = LPool(model, tte)
    pool.add(Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0), -1,
             occ_crop[:, S.PERI:S.PERI + 1] if occ is not None else None)
    pool.set_peri(0)
    hist_idx = np.zeros((n, 0), np.int64); dpre = np.zeros((n, 0)); out = {}; prng = np.random.default_rng(11)
    modes = ["mean", "pw", "pwg", "pwg_shuf"] + (["oracle"] if occ is not None else [])
    need_err = policy in ("epi", "epi_shuf", "ig", "ig_shuf")
    for k in range(Kmax):
        if policy in ("ig", "ig_shuf") and k >= 2:
            wt = ig_weights(dpre, pool, k)
            if policy == "ig_shuf":
                wt = np.stack([rng.permutation(wt[:, j]) for j in range(k)], 1)
            nxt = np.argmax(S.plan_scores(bu, hist_idx, wt, geom), 1)
        else:
            pol = {"ig": "epi", "ig_shuf": "epi"}.get(policy, policy)
            nxt = S.next_fixation(pol, k, bu, hist_idx, dpre, geom, rng, perm, occ_score)
        ids = cr[nxt]; z = np.take_along_axis(Zc, ids[..., None], 1); t = Tc[ids]
        o2n = None
        if need_err:
            ph, o2n = pool.pred_at(t); err = np.mean(np.sum((ph - z) ** 2, -1), 1)
        else:
            err = np.zeros(n)
        pool.add(z, t, k, np.take_along_axis(occ_crop, ids, 1) if occ is not None else None, o2n)
        if k >= 1:
            pool.bump_peri()
        hist_idx = np.concatenate([hist_idx, nxt[:, None]], 1); dpre = np.concatenate([dpre, err[:, None]], 1)
        if k + 1 in ks:
            for md in modes:
                P, wts = pool.predict(md, prng)
                out[(k + 1, "full" if md == "mean" else f"full_{md}")] = {
                    **score(P, zte, yte, knn, per, f"{key}|{k + 1}|{md}"), "n_latents": int(pool.w.sum()), **wdiag(wts, pool, occ is not None)}
            if occ is not None and k + 1 == max(ks):        # does group-LOO error track corruption? (Spearman, per image)
                d = pool.loo_group(); rk = lambda a: np.argsort(np.argsort(a, 1), 1).astype(float)
                rd, ro = rk(d), rk(pool.o); rd -= rd.mean(1, keepdims=True); ro -= ro.mean(1, keepdims=True)
                rho = np.sum(rd * ro, 1) / (np.sqrt(np.sum(rd ** 2, 1) * np.sum(ro ** 2, 1)) + 1e-12)
                out[("diag", "spearman_delta_occ")] = float(np.nanmean(rho))
    c = geom["fix"][hist_idx]
    stats = {"mean_fix_radius": float(np.mean(np.hypot(c[..., 0], c[..., 1]))),
             "revisits": float(np.mean([len(r) - len(set(r)) for r in hist_idx]))}
    if occ is not None:
        sel = np.take_along_axis(occ_score, hist_idx, 1)
        stats["occluded_frac"] = {str(k): float(sel[:, :k].mean()) for k in ks if k <= Kmax}
        stats["occluded_frac_lattice_mean"] = float(occ_score.mean()); stats["occluded_frac_oracle_best"] = float(occ_score.min(1).mean())
    return out, stats, hist_idx


def whole_pools(model, Zv, Tv, Bv, tte, zte, yte, knn, ks, occ, per=None, key=""):
    pool = LPool(model, tte); out = {}; prng = np.random.default_rng(11)
    modes = ["mean", "pw", "pwg_shuf"] + (["oracle"] if occ is not None else [])
    for v in range(max(ks)):
        o = S.occluded_frac(Bv[:, v, 1], Bv[:, v, 0], Bv[:, v, 2], occ, Hh=HH)[:, None] if occ is not None else None
        pool.add(Zv[:, v:v + 1], Tv[:, v:v + 1], -2 - v, o)
        if v + 1 in ks:
            for md in modes:
                P, wts = pool.predict(md, prng)
                out[(v + 1, md)] = {**score(P, zte, yte, knn, per, f"{key}|{v + 1}|{md}"), **wdiag(wts, pool, occ is not None)}
    return out


class GMap:                                     # PC-trained G weights under BP's own inference (review DESIGN F5)
    kind = "Gmap"

    def __init__(self, g):
        self.cfg = dc.replace(g.cfg, rule="bp", start="zero", belief="pc")
        self.p, self.seed = g.p, g.seed
        self.predict = G.make_predict(self.p, None, self.cfg)


def attach_bpg(m, name):
    import equinox as eqx
    d, s = int(name.split("_d")[1].split("_")[0]), int(name.rsplit("_s", 1)[1])
    base = f"pred_runs/tests/gpc_bp_d{d}_zero_bp_s{s}"
    m.cfg = S._cfg_from(json.load(open(base + ".json"))); m.seed = s
    m.p = eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), m.cfg))
    return m


def _bp_predict(m, Zv, Tv, tte, k, batch=250, T=None):
    if m.kind in ("BPg", "Gmap"):               # own inference (joint MAP over the k views)
        pr = m.predict if T is None else G.make_predict(m.p, None, m.cfg, T)
        return np.concatenate([np.asarray(pr(Zv[i:i + batch, :k], Tv[i:i + batch, :k], tte[i:i + batch])) for i in range(0, len(tte), batch)])
    acc = 0                                     # BPf: mean of per-view predictions
    for v in range(k):
        acc = acc + np.concatenate([np.asarray(m.predict(Zv[i:i + batch, v:v + 1], Tv[i:i + batch, v:v + 1], tte[i:i + batch])) for i in range(0, len(tte), batch)])
    return acc / k


def clean_json(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    return x


def dump(res, path):
    tmp = path + ".tmp"; json.dump(clean_json(res), open(tmp, "w"), indent=1, allow_nan=False); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pred_runs/diag_saccade2.json"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=32); ap.add_argument("--vmax", type=int, default=192)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # other datasets (2026-10-30): whole views only, chosen depths, own backbone, no Gmap / convergence diagnostics
    ap.add_argument("--whole-only", action="store_true"); ap.add_argument("--depths", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--backbone", default=S.BACKBONE)
    ap.add_argument("--no-diag", action="store_true")          # drops Gmap only; the BPg T-vs-4T check always runs
    ap.add_argument("--require-all", action="store_true")      # every G, BPg, F, BPf x depth x seed must load
    ap.add_argument("--diag-seed", type=int, default=1)
    a = ap.parse_args()
    global HH, DIAG_SEED
    DIAG_SEED = a.diag_seed
    ks = [k for k in KS if k <= a.kmax]; kv = [k for k in KV if k <= a.vmax]
    L_ = np.load("pred_latents.npz")
    raw_tr = L_["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte, yte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32), np.asarray(L_["yte"])
    knn = KNN((raw_tr - mu) / sd, np.asarray(L_["ytr"]))
    cfg, params = load_run(a.backbone); enc = S.Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32); HH = Xte.shape[1]
    assert a.whole_only or HH == 69, "the saccade front end has Galaxy 69-px constants; use --whole-only (review DS-9)"
    assert Xte.shape[1] == Xte.shape[2] and len(Xte) == len(zte), (Xte.shape, zte.shape)
    if a.n:
        Xte, zte, tte, yte = Xte[:a.n], zte[:a.n], tte[:a.n], yte[:a.n]
    models = {k: v for k, v in S.load_models(a.seeds, a.depths).items()}
    if a.require_all:                        # load_models skips missing BPg silently (Galaxy seed 0); not allowed here
        want = {f"{k}_d10_s{s}" if k in ("F", "BPf") else f"{k}_d{d}_s{s}" for k in ("G", "BPg", "F", "BPf") for d in a.depths for s in a.seeds}
        miss = sorted(want - set(models)); assert not miss, f"missing models: {miss}"
    for name, m in models.items():              # BPg weights/config, for the T vs 4T convergence check
        if m.kind == "BPg":
            attach_bpg(m, name)
    for name in [] if a.no_diag else [k for k in models if k.startswith("G_") and k.endswith(f"_s{DIAG_SEED}")]:   # diagnostic: one seed
        models[name.replace("G_", "Gmap_")] = GMap(models[name])
    geom = S.make_geom("grid4")
    res = {"n": len(zte), "kmax": a.kmax, "vmax": a.vmax, "lam": LAM, "rows": [], "stats": [], "conv": [],
           "ref": {"knn_true_targets": knn(zte, yte), "knn_mean_latent": knn(np.zeros_like(zte) + zte.mean((0, 1)), yte),
                   "sd_min": float(sd.min()), "z_absmax": float(np.abs(zte).max())},
           "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "XLA_FLAGS")}}
    per = {}; npz = a.out.replace(".json", "_perimage.npz")
    for cond in ("occlude", "noise", "clean"):    # occlusion first (rev 3: the full grid did not fit the time limit)
        t0 = time.time()
        Xc, occ = S.corrupt_images(Xte, cond)
        Zc, Tc = (None, None) if a.whole_only else S.crop_table(Xc, enc, cfg.data.img_size)
        views = {v: S.whole_views(Xc, enc, cfg, max(kv), v == "canon", 303 if v == "aug" else 404) for v in ("aug", "canon")}
        print(cond, "encoded", round(time.time() - t0), "s", flush=True)
        for name, m in models.items():
            base = {"model": name, "kind": m.kind, "cond": cond}
            for v, (Zv, Tv, Bv) in views.items():
                if m.kind in ("BPg", "BPf", "Gmap") and v != "aug":   # BP-side rows: augmented views only (rev 3)
                    continue
                if m.kind in ("BPg", "BPf", "Gmap"):     # frozen BP / BP inference: same weights, whole views only
                    for k in kv:
                        P = _bp_predict(m, Zv, Tv, tte, k)
                        res["rows"].append({**base, "input": f"whole_{v}", "policy": "-", "pool": "own", "K": k, "n_latents": k,
                                            **score(P, zte, yte, knn, per, f"{name}|{cond}|whole_{v}|{k}|own")})
                    if m.kind in ("BPg", "Gmap") and v == "aug" and m.seed == DIAG_SEED:        # inference convergence at T vs 4T
                        for k in (48, 192):
                            if k in kv:
                                P4 = _bp_predict(m, Zv, Tv, tte, k, T=4 * m.cfg.T)
                                res["conv"].append({**base, "K": k, "T": m.cfg.T, **{f"T4_{kk}": vv for kk, vv in score(P4, zte, yte, knn).items()}})
                else:
                    for (k, pl), r in whole_pools(m, Zv, Tv, Bv, tte, zte, yte, knn, kv, occ, per, f"{name}|{cond}|whole_{v}").items():
                        res["rows"].append({**base, "input": f"whole_{v}", "policy": "-", "pool": pl, "K": k, "n_latents": k, **r})
            if m.kind not in ("G", "F") or a.whole_only:
                print(cond, name, "whole views done", round(time.time() - t0), "s", flush=True)
                dump(res, a.out); continue
            zp, tp = Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], len(zte), 0)
            P = m.readout(m.embed(zp, tp)[:, 0], tte) if m.kind == "G" else ff(m, zp, tp, tte)[:, 0]
            res["rows"].append({**base, "input": "peripheral", "policy": "-", "pool": "-", "K": 0, "n_latents": 1,
                                **score(P, zte, yte, knn, per, f"{name}|{cond}|peripheral|0")})
            for pol in POLICIES + (["oracle_occ"] if cond == "occlude" else []):
                o, st, hi = run_saccades(m, Zc, Tc, tte, zte, yte, Xc, geom, pol, a.kmax, occ, np.random.default_rng(7 + m.seed),
                                         knn, ks, per, f"{name}|{cond}|saccade|{pol}")
                for (k, pl), r in o.items():
                    if k == "diag":
                        st[pl] = r; continue
                    res["rows"].append({**base, "input": "saccade", "policy": pol, "pool": pl, "K": k, **r})
                res["stats"].append({**base, "policy": pol, **st}); per[f"{name}|{cond}|{pol}|traj"] = hi.astype(np.int16)
                print(cond, name, pol, {pl: round(r["nmse"], 3) for (k, pl), r in o.items() if k == max(ks) and r.get("nmse")},
                      round(time.time() - t0), "s", flush=True)
            dump(res, a.out)
        np.savez_compressed(npz, **per)
    res["per_image_file"] = os.path.basename(npz)
    dump(res, a.out)
    print("DONE")


if __name__ == "__main__":
    main()
