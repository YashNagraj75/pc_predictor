import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
import diag_saccade as S
from diag_saccade import D, load_run

# Follow-up to findings/31, inference only (user 2026-10-28).  Adds, on the grid4 front end of diag_saccade.py:
#  - kNN class probe of the predicted target latents (cosine, k = 20, vs the training images' backbone latents, as
#    metrics.knn_accuracy), for every row;
#  - precision-weighted pooling: weight 1 / (delta_i + LAM median delta), delta_i = leave-one-out error of latent i;
#  - planner 'ig': MPC's planner, each past glimpse weighted by its still-unexplained error times the fraction of its
#    error that later glimpses removed (corrupted regions stop attracting); frozen BP scored on 1..192 whole views.
KS = [1, 2, 4, 8, 16, 32]
KV = [1, 3, 6, 12, 24, 48, 96, 192]
POLICIES = ["random", "epi", "ig", "bu"]
LAM = 0.1
ROWS = 65536


class KNN:
    def __init__(self, ztr, ytr, k=20):
        a = ztr.reshape(-1, ztr.shape[-1]).astype(np.float32)
        self.a = jnp.asarray(a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12))
        self.y = jnp.asarray(np.repeat(ytr, ztr.shape[1])); self.k = k
        self._f = jax.jit(self._pred)

    def _pred(self, b):
        b = b / (jnp.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        _, idx = jax.lax.top_k(b @ self.a.T, self.k)
        return jnp.argmax(jax.nn.one_hot(self.y[idx], 10).sum(1), 1)

    def predict(self, Q):
        return np.concatenate([np.asarray(self._f(jnp.asarray(Q[i:i + 2048], jnp.float32))) for i in range(0, len(Q), 2048)])

    def __call__(self, P, yte):                 # P (n, 4, d) predicted targets -> accuracy over all 4n predictions
        return float(np.mean(self.predict(P.reshape(-1, P.shape[-1])) == np.repeat(yte, P.shape[1])))


def ff(model, z, ta, tb):                       # F on every (latent a, target tau b) pair: (n,a,d),(n,a,9),(n,b,9) -> (n,a,b,d)
    n, A, B = z.shape[0], z.shape[1], tb.shape[1]
    zr = np.repeat(z[:, :, None], B, 2).reshape(-1, z.shape[-1]); tar = np.repeat(ta[:, :, None], B, 2).reshape(-1, 9)
    tbr = np.repeat(tb[:, None], A, 1).reshape(-1, 9)
    out = np.concatenate([model.f(zr[i:i + ROWS], tar[i:i + ROWS], tbr[i:i + ROWS]) for i in range(0, len(zr), ROWS)])
    return out.reshape(n, A, B, -1)


# A set of seen latents with multiplicities w_i (the peripheral latent is counted once per glimpse, as in MPC).
# Per latent: its contribution c_i (G: amortiser embedding; F: predictions of the 4 targets) and, for F,
# R_i = sum_{j != i} w_j f(z_j, tau_j, tau_i) plus the same-glimpse part, so leave-one-out errors cost O(new x all).
class LPool:
    def __init__(self, model, tte):
        self.m, self.tte, self.n = model, tte, len(tte)
        self.Z = self.T = self.C = self.R = self.I = self.Fp = None
        self.w = np.zeros(0); self.g = np.zeros(0, int); self.peri = None

    def _contrib(self, z, t):
        if self.m.kind == "G":
            return self.m.embed(z, t)
        return ff(self.m, z, t, self.tte)       # (n, m, 4, d)

    def pred_at(self, t):                       # current pooled prediction of latents at taus t (n, m, 9)
        if self.m.kind == "G":
            return self.m.readout(self.mean_c(), t)
        return np.einsum("j,njbd->nbd", self.w, ff(self.m, self.Z, self.T, t)) / self.w.sum()

    def mean_c(self, wts=None):
        wts = np.broadcast_to(self.w, (self.n, len(self.w))) if wts is None else wts
        return np.einsum("nj,nj...->n...", wts, self.C) / wts.sum(1).reshape((-1,) + (1,) * (self.C.ndim - 2))

    def add(self, z, t, gid):                   # add m latents of one glimpse (gid >= 0) or one whole view / peripheral
        m = z.shape[1]
        c = self._contrib(z, t)
        if self.m.kind == "F":
            if self.Z is None:
                Rn = np.zeros((self.n, m, z.shape[-1])); In = np.zeros_like(Rn); Fp_new = np.zeros_like(Rn)
            else:
                old_to_new = ff(self.m, self.Z, self.T, t)                     # (n, N, m, d)
                Rn = np.einsum("j,njbd->nbd", self.w, old_to_new)
                new_to_old = ff(self.m, z, t, self.T)                          # (n, m, N, d)
                self.R = self.R + new_to_old.sum(1)
                Fp_new = old_to_new[:, self.peri] if self.peri is not None else np.zeros_like(Rn)
            if m > 1:
                nn = ff(self.m, z, t, t); nn = nn * (1 - np.eye(m))[None, :, :, None]
                In = nn.sum(1); Rn = Rn + In
            else:
                In = np.zeros_like(Rn)
            if self.Z is not None:
                self.I = np.concatenate([self.I, In], 1); self.R = np.concatenate([self.R, Rn], 1)
                self.Fp = np.concatenate([self.Fp, Fp_new], 1)
            else:
                self.I, self.R, self.Fp = In, Rn, Fp_new
        cat = lambda a, b: b if a is None else np.concatenate([a, b], 1)
        self.Z, self.T, self.C = cat(self.Z, z), cat(self.T, t), cat(self.C, c)
        self.w = np.concatenate([self.w, np.ones(m)]); self.g = np.concatenate([self.g, np.full(m, gid)])

    def set_peri(self, idx):
        self.peri = idx
        if self.m.kind == "F":                  # f(z_peri, tau_peri, tau_i) for every latent i (used by bump_peri)
            self.Fp = ff(self.m, self.Z[:, idx:idx + 1], self.T[:, idx:idx + 1], self.T)[:, 0]
            self.Fp[:, idx] = 0.0

    def bump_peri(self):                        # peripheral latent seen once more
        self.w[self.peri] += 1
        if self.m.kind == "F":
            self.R = self.R + self.Fp

    def loo(self, idx=None):                    # leave-one-out squared error of every latent, (n, N)
        idx = np.arange(len(self.w)) if idx is None else idx
        if self.m.kind == "G":
            S_ = np.einsum("j,njm->nm", self.w, self.C); W = self.w.sum()
            s = (S_[:, None] - self.w[idx][None, :, None] * self.C[:, idx]) / (W - self.w[idx])[None, :, None]
            zh = self.m.readout(s.reshape(-1, s.shape[-1]), self.T[:, idx].reshape(-1, 1, 9))[:, 0].reshape(self.n, len(idx), -1)
        else:
            zh = self.R[:, idx] / (self.w.sum() - self.w[idx])[None, :, None]
        return np.sum((zh - self.Z[:, idx]) ** 2, -1)

    def glimpse_err(self, gid):                 # error of glimpse gid's latents with the whole glimpse left out, (n,)
        idx = np.where(self.g == gid)[0]; W = self.w.sum() - self.w[idx].sum()
        if self.m.kind == "G":
            S_ = np.einsum("j,njm->nm", self.w, self.C) - np.einsum("j,njm->nm", self.w[idx], self.C[:, idx])
            zh = self.m.readout(S_ / W, self.T[:, idx])
        else:
            zh = (self.R[:, idx] - self.I[:, idx]) / W
        return np.mean(np.sum((zh - self.Z[:, idx]) ** 2, -1), 1)

    def precision(self):
        if len(self.w) < 2:
            return np.ones((self.n, len(self.w)))
        d = self.loo()
        return 1.0 / (d + LAM * np.median(d, 1, keepdims=True) + 1e-12)

    def predict(self, weighted=False):
        wts = self.w[None] * self.precision() if weighted else None
        c = self.mean_c(wts)
        return self.m.readout(c, self.tte) if self.m.kind == "G" else c


def ig_weights(dpre, pool, k):                  # weights for S_TD over glimpses 0..k-1
    w = np.zeros_like(dpre)
    for j in range(k):
        if j == k - 1:
            w[:, j] = dpre[:, j]                # latest glimpse: no later evidence yet (MPC's weight)
        else:
            e = pool.glimpse_err(j)
            w[:, j] = e * np.clip((dpre[:, j] - e) / (dpre[:, j] + 1e-12), 0.0, 1.0)
    return w


def score(P, zte, yte, knn):
    m, _ = S.metrics(P, zte)
    return {"nmse": m["nmse"], "knn": knn(P, yte)}


def run_saccades(model, Zc, Tc, tte, zte, yte, X, geom, policy, Kmax, occ, rng, knn, ks=None):
    ks = ks or KS
    n, L, cr = len(zte), geom["L"], geom["crops"]
    bu = S.bottom_up(X, geom)
    perm = rng.permuted(np.tile(np.arange(L), (n, 1)), axis=1)
    occ_score = None
    if occ is not None:
        sp = S.crop_specs()
        occ_score = np.stack([np.mean([S.occluded_frac(sp[c][0], sp[c][1], sp[c][2], occ) for c in cr[l]], 0) for l in range(L)], 1)
    pool = LPool(model, tte)
    pool.add(Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0), -1); pool.set_peri(0)
    hist_idx = np.zeros((n, 0), np.int64); dpre = np.zeros((n, 0)); out = {}
    for k in range(Kmax):
        if policy == "ig" and k >= 2:
            nxt = np.argmax(S.plan_scores(bu, hist_idx, ig_weights(dpre, pool, k), geom), 1)
        else:
            pol = "epi" if policy == "ig" else policy
            nxt = S.next_fixation(pol, k, bu, hist_idx, dpre, geom, rng, perm, occ_score)
        ids = cr[nxt]; z = np.take_along_axis(Zc, ids[..., None], 1); t = Tc[ids]
        err = np.mean(np.sum((pool.pred_at(t) - z) ** 2, -1), 1) if policy in ("epi", "ig") else np.zeros(n)
        pool.add(z, t, k)
        if k >= 1:
            pool.bump_peri()
        hist_idx = np.concatenate([hist_idx, nxt[:, None]], 1); dpre = np.concatenate([dpre, err[:, None]], 1)
        if k + 1 in ks:
            for wtd in (False, True):
                out[(k + 1, "full_pw" if wtd else "full")] = {**score(pool.predict(wtd), zte, yte, knn), "n_latents": int(pool.w.sum())}
    stats = {}
    if occ is not None:
        sel = np.take_along_axis(occ_score, hist_idx, 1)
        stats["occluded_frac"] = {str(k): float(sel[:, :k].mean()) for k in ks if k <= Kmax}
    return out, stats, hist_idx


def whole_pools(model, Zv, Tv, tte, zte, yte, knn, ks):    # PC: mean and precision-weighted pools of whole views
    pool = LPool(model, tte); out = {}
    for v in range(max(ks)):
        pool.add(Zv[:, v:v + 1], Tv[:, v:v + 1], -2 - v)
        if v + 1 in ks:
            for wtd in (False, True):
                out[(v + 1, "pw" if wtd else "mean")] = score(pool.predict(wtd), zte, yte, knn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pred_runs/diag_saccade2.json"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=32); ap.add_argument("--vmax", type=int, default=192)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    ks = [k for k in KS if k <= a.kmax]; kv = [k for k in KV if k <= a.vmax]
    L_ = np.load("pred_latents.npz")
    raw_tr = L_["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte, yte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32), np.asarray(L_["yte"])
    knn = KNN((raw_tr - mu) / sd, np.asarray(L_["ytr"]))
    cfg, params = load_run(S.BACKBONE); enc = S.Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32)
    if a.n:
        Xte, zte, tte, yte = Xte[:a.n], zte[:a.n], tte[:a.n], yte[:a.n]
    models = S.load_models(a.seeds); geom = S.make_geom("grid4")
    res = {"n": len(zte), "kmax": a.kmax, "vmax": a.vmax, "lam": LAM, "rows": [], "stats": [],
           "ref": {"knn_true_targets": knn(zte, yte)},
           "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "XLA_FLAGS")}}
    for cond in ("clean", "noise", "occlude"):
        t0 = time.time()
        Xc, occ = S.corrupt_images(Xte, cond)
        Zc, Tc = S.crop_table(Xc, enc, cfg.data.img_size)
        views = {v: S.whole_views(Xc, enc, cfg, max(kv), v == "canon", 303 if v == "aug" else 404) for v in ("aug", "canon")}
        print(cond, "encoded", round(time.time() - t0), "s", flush=True)
        for name, m in models.items():
            base = {"model": name, "kind": m.kind, "cond": cond}
            for v, (Zv, Tv, _) in views.items():
                if m.kind.startswith("BP"):     # frozen BP, same weights, whole views only (user 2026-10-28)
                    for k in kv:
                        P = _bp_predict(m, Zv, Tv, tte, k)
                        res["rows"].append({**base, "input": f"whole_{v}", "policy": "-", "pool": "mean", "K": k, "n_latents": k, **score(P, zte, yte, knn)})
                else:
                    for (k, pl), r in whole_pools(m, Zv, Tv, tte, zte, yte, knn, kv).items():
                        res["rows"].append({**base, "input": f"whole_{v}", "policy": "-", "pool": pl, "K": k, "n_latents": k, **r})
            if m.kind.startswith("BP"):
                print(cond, name, "whole views done", round(time.time() - t0), "s", flush=True)
                continue
            P = (m.readout(m.embed(Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], len(zte), 0))[:, 0], tte)
                 if m.kind == "G" else ff(m, Zc[:, S.PERI:S.PERI + 1], np.repeat(Tc[None, S.PERI:S.PERI + 1], len(zte), 0), tte)[:, 0])
            res["rows"].append({**base, "input": "peripheral", "policy": "-", "pool": "-", "K": 0, "n_latents": 1, **score(P, zte, yte, knn)})
            for pol in POLICIES + (["oracle_occ"] if cond == "occlude" else []):
                o, st, hi = run_saccades(m, Zc, Tc, tte, zte, yte, Xc, geom, pol, a.kmax, occ, np.random.default_rng(7 + m.seed), knn, ks)
                for (k, pl), r in o.items():
                    res["rows"].append({**base, "input": "saccade", "policy": pol, "pool": pl, "K": k, **r})
                res["stats"].append({**base, "policy": pol, **st})
                print(cond, name, pol, {f"{k}{pl[-2:]}": round(r["nmse"], 3) for (k, pl), r in o.items() if k in (1, max(ks))}, round(time.time() - t0), "s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


def _bp_predict(m, Zv, Tv, tte, k, batch=250):  # BPg: its own inference from k views; BPf: mean of per-view predictions
    if m.kind == "BPg":
        return np.concatenate([np.asarray(m.predict(Zv[i:i + batch, :k], Tv[i:i + batch, :k], tte[i:i + batch])) for i in range(0, len(tte), batch)])
    acc = 0
    for v in range(k):
        acc = acc + np.concatenate([np.asarray(m.predict(Zv[i:i + batch, v:v + 1], Tv[i:i + batch, v:v + 1], tte[i:i + batch])) for i in range(0, len(tte), batch)])
    return acc / k


if __name__ == "__main__":
    main()
