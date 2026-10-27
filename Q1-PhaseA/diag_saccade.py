import argparse, dataclasses as dc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from scipy.ndimage import gaussian_filter
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, pred_eval as PE, pred_gpc as G, pred_ff as PF, pred_ffres as RF
from pred_data import tau_features
from run_probes import load_run

# S2 of the saccade plan (wiki 'Method - Saccades for the PC predictors plan', findings/30): MPC-style glimpses fed to
# the trained PC predictors, inference only.  BP references are NOT given glimpses (user rule 2026-10-27); they are
# only re-scored on one fresh whole augmented view under the same image-space corruption.
# Glimpse at fixation c (lattice index i, j): 4 foveal crops (area 0.4) at c +- 2h in a 2x2 grid, 1 parafoveal crop
# (area 0.6) at c clipped to its range, 1 peripheral crop (area 1, centred; the same latent every glimpse, as in MPC).
# Aggregation = MPC's mean over every crop latent seen (G: mean amortised belief; F: mean of per-crop predictions).
# Planner = MPC _plan_next_saccade (lgn_glimpser.py) on the 69x69 image, scored at lattice points.
BACKBONE = "runs/bp_sigreg_etafix_s0"
A_F, A_P = 0.4, 0.6
S_F, S_P = float(np.sqrt(A_F)), float(np.sqrt(A_P))
H_STEP = 0.06                                   # lattice spacing; fixations i*h, |i| <= 4; foveal offset 2h
FIX = np.arange(-4, 5)                          # 9 x 9 fixation lattice, |c| <= 0.24
FOV = np.arange(-6, 7)                          # 13 x 13 foveal-crop lattice, |c| <= 0.36 <= 1 - S_F
LIM_P = 1.0 - S_P
SCALE = 69.0 / 28.0                             # MPC's pixel sigmas were set on 28 x 28 MNIST
SIG_BU, SIG_ERR, SIG_IOR, BETA = 8.0 * SCALE, 7.5 * SCALE, 5.0 * SCALE, 0.5
KS = [1, 2, 4, 8, 16, 32]
KV = [1, 2, 4, 8, 16]


# ------------------------------------------------------------------ crops
def _render(img, cy, cx, s, out):
    g = jnp.linspace(-1.0, 1.0, out)
    yy, xx = jnp.meshgrid(g, g, indexing="ij")
    H, W, C = img.shape
    px = (s * xx + cx + 1.0) * 0.5 * (W - 1)
    py = (s * yy + cy + 1.0) * 0.5 * (H - 1)
    v = jnp.stack([D.map_coordinates(img[..., c], [py, px], order=1, mode="nearest") for c in range(C)], -1)
    return jnp.clip((v - 0.5) * 1.0 + 0.5 + 0.0, 0.0, 1.0)     # same colour step as data._one_view at zero jitter


render = jax.jit(jax.vmap(_render, in_axes=(0, None, None, None, None)), static_argnums=4)


def crop_tau(cy, cx, s):
    return tau_features(np.array([cx, cy, s, s], np.float32), np.zeros(4, np.float32))


def to_vec(v, cfg):
    if cfg.data.grayscale and v.shape[-1] == 3:
        v = v.mean(-1, keepdims=True)
    return np.asarray(v).reshape(v.shape[0], -1)


# All crop latents for one image set (normalised): Zf (n,13,13,d), Zp (n,9,9,d), Zw (n,d); taus alike.
def lattice(X, cfg, params, mu, sd, chunk=250):
    enc = lambda pix: (PE.encode(params, cfg.enc, pix[:, None])[:, 0] - mu) / sd
    n = len(X); Zf = np.zeros((n, 13, 13, 128), np.float32); Zp = np.zeros((n, 9, 9, 128), np.float32)
    Zw = np.zeros((n, 128), np.float32)
    for i0 in range(0, n, chunk):
        xb = jnp.asarray(X[i0:i0 + chunk])
        for a, ia in enumerate(FOV):
            for b, ib in enumerate(FOV):
                Zf[i0:i0 + chunk, a, b] = enc(to_vec(render(xb, ia * H_STEP, ib * H_STEP, S_F, cfg.data.img_size), cfg))
        for a, ia in enumerate(FIX):
            for b, ib in enumerate(FIX):
                cy, cx = np.clip([ia * H_STEP, ib * H_STEP], -LIM_P, LIM_P)
                Zp[i0:i0 + chunk, a, b] = enc(to_vec(render(xb, cy, cx, S_P, cfg.data.img_size), cfg))
        Zw[i0:i0 + chunk] = enc(to_vec(render(xb, 0.0, 0.0, 1.0, cfg.data.img_size), cfg))
    Tf = np.array([[crop_tau(ia * H_STEP, ib * H_STEP, S_F) for ib in FOV] for ia in FOV], np.float32)
    Tp = np.array([[crop_tau(*np.clip([ia * H_STEP, ib * H_STEP], -LIM_P, LIM_P), S_P) for ib in FIX] for ia in FIX], np.float32)
    return Zf, Zp, Zw, Tf, Tp, crop_tau(0.0, 0.0, 1.0)


# Crops of glimpse (i, j) (lattice indices into FIX): list of (kind, a, b), peripheral excluded (handled by weight).
def glimpse_crops(i, j):
    fi, fj = i + 2, j + 2                    # FIX index -> FOV index of the fixation itself (FIX[-4] = FOV[-4] = index 2)
    return [("f", fi - 2, fj - 2), ("f", fi - 2, fj + 2), ("f", fi + 2, fj - 2), ("f", fi + 2, fj + 2), ("p", i, j)]


# Vectorised glimpse_crops for every image at once (same order: 4 foveal, then parafoveal).
def gather(Zf, Zp, Tf, Tp, gi, gj):
    r = np.arange(len(gi)); fi, fj = gi + 2, gj + 2
    idx = [(fi - 2, fj - 2), (fi - 2, fj + 2), (fi + 2, fj - 2), (fi + 2, fj + 2)]
    zc = np.stack([Zf[r, a, b] for a, b in idx] + [Zp[r, gi, gj]], 1)
    tc = np.stack([Tf[a, b] for a, b in idx] + [Tp[gi, gj]], 1)
    return zc, tc


# Fresh augmented whole views (the training distribution), key 303, V per image.
def fresh_views(X, cfg, params, mu, sd, V, key=303, chunk=500):
    one = jax.vmap(lambda k, im: D._one_view(k, im, cfg.data.img_size, cfg.data.crop_area[0], cfg.data.crop_area[1],
                                             cfg.data.max_rotation, cfg.data.hflip, cfg.data.brightness, cfg.data.contrast))
    Z, T = [], []
    for i0 in range(0, len(X), chunk):
        xb = jnp.asarray(X[i0:i0 + chunk]); zs, ts = [], []
        for v in range(V):
            ks = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(key), i0 * 100 + v), len(xb))
            out, bx, pr = one(ks, xb)
            zs.append((PE.encode(params, cfg.enc, to_vec(out, cfg)[:, None])[:, 0] - mu) / sd); ts.append(tau_features(bx, pr))
        Z.append(np.stack(zs, 1)); T.append(np.stack(ts, 1))
    return np.concatenate(Z), np.concatenate(T)


# ------------------------------------------------------------------ image-space corruption
def corrupt_images(X, kind, seed=0):
    rng = np.random.default_rng(seed); X = X.copy(); n, Hh, Ww, _ = X.shape
    occ = np.zeros((n, 4), np.int64)
    if kind == "noise":
        X = np.clip(X + 0.2 * rng.standard_normal(X.shape).astype(np.float32), 0.0, 1.0)
    elif kind == "occlude":
        s = int(round(Hh * np.sqrt(0.25)))
        r0, c0 = rng.integers(0, Hh - s + 1, n), rng.integers(0, Ww - s + 1, n)
        for i in range(n):
            X[i, r0[i]:r0[i] + s, c0[i]:c0[i] + s] = 0.0
        occ = np.stack([r0, c0, r0 + s, c0 + s], 1)
    return X, occ


def occluded_frac(cy, cx, s, occ, Hh=69):
    lo_r, hi_r = (cy - s + 1) * 0.5 * (Hh - 1), (cy + s + 1) * 0.5 * (Hh - 1)
    lo_c, hi_c = (cx - s + 1) * 0.5 * (Hh - 1), (cx + s + 1) * 0.5 * (Hh - 1)
    ir = np.clip(np.minimum(hi_r, occ[:, 2]) - np.maximum(lo_r, occ[:, 0]), 0, None)
    ic = np.clip(np.minimum(hi_c, occ[:, 3]) - np.maximum(lo_c, occ[:, 1]), 0, None)
    return ir * ic / ((hi_r - lo_r) * (hi_c - lo_c))


# ------------------------------------------------------------------ MPC planner (lgn_glimpser._plan_next_saccade)
# Scored only at lattice points; S_BU and S_TD are each scaled to max 1 per image (MPC mixes raw scales; ours differ).
LAT_PIX = (FIX * H_STEP + 1.0) * 0.5 * 68.0
LY, LX = np.meshgrid(LAT_PIX, LAT_PIX, indexing="ij"); LY, LX = LY.ravel(), LX.ravel()


def bottom_up(X):
    e = np.abs(X).sum(-1)
    bu = np.stack([gaussian_filter(m, SIG_BU, mode="nearest") for m in e])
    iy, ix = np.rint(LY).astype(int), np.rint(LX).astype(int)
    b = bu[:, iy, ix]
    return b / (b.max(1, keepdims=True) + 1e-12)                           # (n, 81)


def plan(bu, hist_idx, hist_err, beta=BETA):
    n = bu.shape[0]
    if hist_idx.shape[1] == 0:
        return np.argmax(bu, 1)
    hy, hx = LY[hist_idx], LX[hist_idx]                                      # (n, k)
    d2 = (LY[None, :, None] - hy[:, None]) ** 2 + (LX[None, :, None] - hx[:, None]) ** 2   # (n, 81, k)
    td = np.sum(hist_err[:, None] * np.exp(-0.5 * d2 / SIG_ERR ** 2), -1)
    td = td / (td.max(1, keepdims=True) + 1e-12)
    ior = np.prod(1.0 - np.exp(-0.5 * d2 / SIG_IOR ** 2), -1)
    return np.argmax(((1 - beta) * bu + beta * td) * ior, 1)


# ------------------------------------------------------------------ predictors (sum-and-count form of MPC's mean)
class GPred:
    def __init__(self, depth, seed, root="pred_runs/tests_sep_fix"):
        base = f"{root}/gpc_sep_d{depth}_amort_hp16_a1_k1_amort_c1_T2200_bp_s{seed}"
        d = json.load(open(base + ".json"))
        names = {f.name for f in dc.fields(G.GCfg)}
        self.cfg = G.GCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["cfg"].items() if k in names})
        self.p = eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), self.cfg))
        self.am = eqx.tree_deserialise_leaves(base + "_amort.eqx", G.init_amort(jax.random.PRNGKey(0)))
        self.k1_stored = next(r["nmse"] for r in d["rows"] if r["test"] == "k_seen" and r.get("k") == 1)
        H = self.cfg.H
        self._a = eqx.filter_jit(lambda am, z, t: PF.forward(am, jnp.concatenate([z, t], -1)))
        self._d = eqx.filter_jit(lambda p, s, t: G.topdown(p, s, G.phi(t), H)[1])
        self.predict = G.make_predict(self.p, self.am, self.cfg)

    def amort(self, z, t):                       # (N, d), (N, 9) -> (N, M)
        return np.asarray(self._a(self.am, jnp.asarray(z), jnp.asarray(t)))

    def decode(self, s, t):                      # (B, M), (B, m, 9) -> (B, m, d)
        return np.asarray(self._d(self.p, jnp.asarray(s), jnp.asarray(t)))


class FPred:
    def __init__(self, seed, arm="R16"):
        rc = RF.RCfg(arm="alm", depth=10, seed=seed, T=400, alpha=1.0, lr=0.01, rho=16.0)
        self.params = RF.load_model(f"pred_runs/ffres_fix/ffres_alm{arm}_d10_s{seed}.eqx", rc)
        self.px = RF.make_predict_x(self.params, rc)
        self.predict = RF.make_pair_predict(self.params, rc)

    def f(self, z, ta, tb):                      # rows (N, d), (N, 9), (N, 9) -> (N, d)
        return self.px(np.concatenate([z, ta, tb], -1))


def metrics(P, zte):
    n = len(zte); img = np.repeat(np.arange(n), 4); tv = np.tile(np.arange(4), n)
    return PE._metrics(P.reshape(-1, P.shape[-1]), zte.reshape(-1, zte.shape[-1]), img, tv, zte)


# One saccade trajectory for every image: returns {K: metrics}, fixation statistics, the first 5 trajectories.
# Surprise of a new crop = error of predicting it from every OTHER crop seen so far (MPC's per-glimpse error delta).
def run_saccades(model, kind, Lt, tte, zte, bu, policy, Kmax, occ, rng, batch=200):
    Zf, Zp, Zw, Tf, Tp, Tw = Lt
    n = len(zte); out = {}
    Tw_n = np.repeat(Tw[None], n, 0)
    perm = rng.permuted(np.tile(np.arange(81), (n, 1)), axis=1)[:, :Kmax]
    hist_idx = np.zeros((n, 0), np.int64); hist_err = np.zeros((n, 0)); occ_fr = []
    if kind == "G":
        aw = model.amort(Zw, Tw_n); S = np.zeros_like(aw)
    else:
        fw = np.stack([model.f(Zw, Tw_n, tte[:, t]) for t in range(4)], 1); S = np.zeros_like(fw)
        Zs_all, Ts_all = np.zeros((n, 0, 128), np.float32), np.zeros((n, 0, 9), np.float32)
    N = 0
    for k in range(1, Kmax + 1):
        nxt = perm[:, k - 1] if policy == "random" else plan(bu, hist_idx, hist_err)
        gi, gj = nxt // 9, nxt % 9
        zc, tc = gather(Zf, Zp, Tf, Tp, gi, gj)                      # (n, 5, d), (n, 5, 9)
        if occ is not None:
            cy, cx = FIX[gi] * H_STEP, FIX[gj] * H_STEP
            occ_fr.append(np.mean([occluded_frac(cy + dy, cx + dx, S_F, occ) for dy in (-2 * H_STEP, 2 * H_STEP)
                                   for dx in (-2 * H_STEP, 2 * H_STEP)], 0))
        N += 6                                                        # 4 foveal + 1 parafoveal + the peripheral again
        if kind == "G":
            ac = model.amort(zc.reshape(-1, 128), tc.reshape(-1, 9)).reshape(n, 5, -1)
            S = S + ac.sum(1) + aw
            s_loo = (S[:, None] - ac) / (N - 1)
            zh = model.decode(s_loo.reshape(n * 5, -1), tc.reshape(n * 5, 1, 9))[:, 0].reshape(n, 5, -1)
            err = np.mean(np.sum((zh - zc) ** 2, -1), 1)
        else:
            fc = np.stack([np.stack([model.f(zc[:, v], tc[:, v], tte[:, t]) for v in range(5)], 1) for t in range(4)], 1)
            S = S + fc.sum(2) + fw
            Zs_all, Ts_all = np.concatenate([Zs_all, zc], 1), np.concatenate([Ts_all, tc], 1)
            err = np.zeros(n)
            if policy == "epi":
                No = Zs_all.shape[1]
                for v in range(5):
                    tv = np.repeat(tc[:, v:v + 1], No, 1)
                    pr = model.f(Zs_all.reshape(-1, 128), Ts_all.reshape(-1, 9), tv.reshape(-1, 9)).reshape(n, No, -1)
                    own = No - 5 + v                                  # column of crop v itself
                    tot = pr.sum(1) - pr[:, own] + k * model.f(Zw, Tw_n, tc[:, v])
                    err += np.sum((tot / (N - 1) - zc[:, v]) ** 2, -1) / 5
        hist_idx = np.concatenate([hist_idx, nxt[:, None]], 1); hist_err = np.concatenate([hist_err, err[:, None]], 1)
        if k in KS:
            P = (np.concatenate([model.decode(S[i:i + batch] / N, tte[i:i + batch]) for i in range(0, n, batch)])
                 if kind == "G" else S / N)
            out[k] = {**metrics(P, zte), "n_latents": N}
    stats = {"mean_fix_radius": float(np.mean(np.hypot(FIX[hist_idx // 9], FIX[hist_idx % 9]) * H_STEP)),
             "revisits": float(np.mean([len(r) - len(set(r)) for r in hist_idx])),
             "first_fix_centre_frac": float(np.mean(hist_idx[:, 0] == 40))}
    if occ_fr:
        o = np.stack(occ_fr, 1); stats["foveal_occluded_frac"] = {str(k): float(o[:, :k].mean()) for k in KS}
    return out, stats, hist_idx[:5].tolist()


def whole_views(model, kind, Zv, Tv, tte, zte, ks, batch=200):
    out = {}
    for k in ks:
        if kind in ("G", "BPg"):
            P = np.concatenate([np.asarray(model.predict(Zv[i:i + batch, :k], Tv[i:i + batch, :k], tte[i:i + batch])) for i in range(0, len(zte), batch)])
        else:
            P = np.mean([np.concatenate([np.asarray(model.predict(Zv[i:i + batch, v:v + 1], Tv[i:i + batch, v:v + 1], tte[i:i + batch]))
                                         for i in range(0, len(zte), batch)]) for v in range(k)], 0)
        out[k] = metrics(P, zte)
    return out


class BPg:                                       # frozen BP auto-decoder G1: whole view only
    def __init__(self, depth, seed):
        base = f"pred_runs/tests/gpc_bp_d{depth}_zero_bp_s{seed}"
        d = json.load(open(base + ".json"))
        names = {f.name for f in dc.fields(G.GCfg)}
        cfg = G.GCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["cfg"].items() if k in names})
        p = eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), cfg))
        self.predict = G.make_predict(p, None, cfg)


class BPf:                                       # frozen residual BP d10: whole view only
    def __init__(self, seed):
        rc = RF.RCfg(arm="bp", depth=10, seed=seed, lr=0.001)
        self.predict = RF.make_pair_predict(RF.load_model(f"pred_runs/frozen_bp/ffres_bp_d10_s{seed}.eqx", rc), rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pred_runs/diag_saccade.json"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=32); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    global KS, KV
    KS = [k for k in KS if k <= a.kmax]; KV = [k for k in KV if k <= min(a.kmax, 16)]
    L = np.load("pred_latents.npz")
    raw_tr = L["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (L["bp_emb_te"] - mu) / sd, L["tau_te"].astype(np.float32)
    cfg, params = load_run(BACKBONE)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32)
    if a.n:
        Xte, zte, tte = Xte[:a.n], zte[:a.n], tte[:a.n]
    models = {}
    for s in a.seeds:
        for d in (3, 10):
            models[f"G_d{d}_s{s}"] = ("G", GPred(d, s))
            models[f"BPg_d{d}_s{s}"] = ("BPg", BPg(d, s))
        models[f"F_d10_s{s}"] = ("F", FPred(s)); models[f"BPf_d10_s{s}"] = ("BPf", BPf(s))
    res = {"backbone": BACKBONE, "n": len(zte), "kmax": a.kmax, "rows": [], "stats": [], "traj": {},
           "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "XLA_FLAGS")},
           "settings": {"A_F": A_F, "A_P": A_P, "h": H_STEP, "sig_bu": SIG_BU, "sig_err": SIG_ERR, "sig_ior": SIG_IOR, "beta": BETA}}
    for cond in ("clean", "noise", "occlude"):
        t0 = time.time()
        Xc, occ = corrupt_images(Xte, cond)
        Lt = lattice(Xc, cfg, params, mu, sd)
        Zv, Tv = fresh_views(Xc, cfg, params, mu, sd, max(KV))
        bu = bottom_up(Xc)
        print(cond, "encoded", round(time.time() - t0), "s", flush=True)
        for name, (kind, m) in models.items():
            ks = [1] if kind.startswith("BP") else KV
            for k, r in whole_views(m, kind, Zv, Tv, tte, zte, ks).items():
                res["rows"].append({"model": name, "kind": kind, "cond": cond, "input": "whole", "policy": "-", "K": k, **r})
            if kind in ("G", "F"):
                for pol in ("random", "epi"):
                    o, st, tr = run_saccades(m, kind, Lt, tte, zte, bu, pol, a.kmax, occ if cond == "occlude" else None,
                                             np.random.default_rng(7))
                    for k, r in o.items():
                        res["rows"].append({"model": name, "kind": kind, "cond": cond, "input": "saccade", "policy": pol, "K": k, **r})
                    res["stats"].append({"model": name, "cond": cond, "policy": pol, **st}); res["traj"][f"{name}/{cond}/{pol}"] = tr
                    print(cond, name, pol, {k: round(v["nmse"], 4) for k, v in o.items()}, st, round(time.time() - t0), "s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
