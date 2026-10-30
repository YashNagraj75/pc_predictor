import argparse, dataclasses as dc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, encoder as E, pred_eval as PE, pred_gpc as G, pred_ff as PF, pred_ffres as RF
from pred_data import tau_features
from run_probes import load_run

# S2 of the saccade plan (wiki 'Method - Saccades for the PC predictors plan'; findings/30), revision 2 after 3 reviews.
# MPC-style glimpses fed to the trained PC predictors, inference only.  BP references get NO glimpses (user rule
# 2026-10-27): one augmented whole view, re-scored under the same image-space corruption.
# Crop table per image: foveal crops (area 0.4) and parafoveal crops (area 0.6, centre clipped) on a 13 x 13 lattice
# (spacing h, |c| <= 6h = 0.36), plus the peripheral whole-image crop (area 1, centred).  All crops unrotated, unjittered.
# Geometries: 'grid4' = MPC's 2x2 foveal grid at c +- 2h + parafoveal + peripheral, fixations |c| <= 4h (9 x 9);
#             'fov1'  = one foveal crop at c + parafoveal + peripheral, fixations on the full 13 x 13 lattice (wider).
# Aggregation = MPC's mean over the crop latents in a pool.  Planner = MPC _plan_next_saccade, scored at lattice points.
BACKBONE = "runs/bp_sigreg_etafix_s0"
A_F, A_P = 0.4, 0.6
S_F, S_P = float(np.sqrt(A_F)), float(np.sqrt(A_P))
H_STEP = 0.06
LAT = np.arange(-6, 7)                          # 13 x 13 crop lattice
LIM_P = 1.0 - S_P
SCALE = 69.0 / 28.0                             # MPC's pixel sigmas were set on 28 x 28 MNIST
SIG_BU, SIG_ERR, SIG_IOR, BETA = 8.0 * SCALE, 7.5 * SCALE, 5.0 * SCALE, 0.5
KS = [1, 2, 4, 8, 16, 32]
KV = [1, 3, 6, 12, 24, 48, 96, 192]             # whole-view counts: exact-N partners of the saccade pools
POOLS = ["full", "noperi", "fov", "para"]
POLICIES = ["random", "epi", "epi_shuf", "ior", "bu"]
NCROP = 2 * 169 + 1                            # foveal 0..168, parafoveal 169..337, peripheral 338
PERI = NCROP - 1


# ------------------------------------------------------------------ geometry
def lat_index(a, b):                            # lattice offsets a, b in -6..6 -> crop row in the 13 x 13 table
    return (a + 6) * 13 + (b + 6)


def make_geom(name):
    if name == "grid4":
        fx = [(a, b) for a in range(-4, 5) for b in range(-4, 5)]
        crops = [[lat_index(a - 2, b - 2), lat_index(a - 2, b + 2), lat_index(a + 2, b - 2), lat_index(a + 2, b + 2),
                  169 + lat_index(a, b)] for a, b in fx]
        kinds = ["f", "f", "f", "f", "p"]
    else:
        fx = [(a, b) for a in range(-6, 7) for b in range(-6, 7)]
        crops = [[lat_index(a, b), 169 + lat_index(a, b)] for a, b in fx]
        kinds = ["f", "p"]
    fx = np.array(fx); c = fx * H_STEP
    return {"name": name, "fix": c, "crops": np.array(crops), "kinds": kinds,
            "py": (c[:, 0] + 1.0) * 0.5 * 68.0, "px": (c[:, 1] + 1.0) * 0.5 * 68.0, "L": len(fx)}


def crop_specs():                               # (cy, cx, s) per crop-table row
    sp = [(a * H_STEP, b * H_STEP, S_F) for a in LAT for b in LAT]
    sp += [tuple(np.clip([a * H_STEP, b * H_STEP], -LIM_P, LIM_P)) + (S_P,) for a in LAT for b in LAT]
    return np.array(sp + [(0.0, 0.0, 1.0)], np.float32)


def crop_tau(cy, cx, s):
    return tau_features(np.array([cx, cy, s, s], np.float32), np.zeros(4, np.float32))


# ------------------------------------------------------------------ rendering and encoding
def _render(img, cy, cx, s, out):
    g = jnp.linspace(-1.0, 1.0, out)
    yy, xx = jnp.meshgrid(g, g, indexing="ij")
    H, W, C = img.shape
    px = (s * xx + cx + 1.0) * 0.5 * (W - 1)
    py = (s * yy + cy + 1.0) * 0.5 * (H - 1)
    v = jnp.stack([D.map_coordinates(img[..., c], [py, px], order=1, mode="nearest") for c in range(C)], -1)
    return jnp.clip((v - 0.5) * 1.0 + 0.5 + 0.0, 0.0, 1.0)     # data._one_view's colour step at zero jitter


render = jax.jit(jax.vmap(_render, in_axes=(0, None, None, None, None)), static_argnums=4)


class Encoder:                                  # one compiled forward, fixed chunk (review CORR F8)
    def __init__(self, cfg, params, mu, sd, chunk=1024):
        self.cfg, self.p, self.mu, self.sd, self.chunk = cfg, params, mu, sd, chunk
        self.f = eqx.filter_jit(lambda p, x: E.forward(p, x, cfg.enc)[0])

    def __call__(self, pix):                    # pix (N, H, W, C) in [0, 1] -> normalised latents (N, d)
        if self.cfg.data.grayscale and pix.shape[-1] == 3:
            pix = pix.mean(-1, keepdims=True)
        x = np.asarray(pix).reshape(len(pix), -1); out = []
        for i in range(0, len(x), self.chunk):
            b = x[i:i + self.chunk]; m = len(b)
            if m < self.chunk:
                b = np.concatenate([b, np.zeros((self.chunk - m, b.shape[1]), b.dtype)])
            out.append(np.asarray(self.f(self.p, jnp.asarray(b)))[:m])
        return (np.concatenate(out) - self.mu) / self.sd


def crop_table(X, enc, size, chunk=1024):        # (n, NCROP, d) latents, (NCROP, 9) taus
    sp = crop_specs(); n = len(X); Z = np.zeros((n, NCROP, 128), np.float32)
    for i0 in range(0, n, chunk):
        xb = jnp.asarray(X[i0:i0 + chunk])
        for r, (cy, cx, s) in enumerate(sp):
            Z[i0:i0 + chunk, r] = enc(render(xb, float(cy), float(cx), float(s), size))
    return Z, np.stack([crop_tau(*r) for r in sp]).astype(np.float32)


# V whole views per image: 'aug' = the training augmentation; 'canon' = same crop range, no rotation/flip/jitter.
def whole_views(X, enc, cfg, V, canon, key, chunk=500):
    dd = cfg.data
    rot, flip, br, ct = (0.0, False, 0.0, 0.0) if canon else (dd.max_rotation, dd.hflip, dd.brightness, dd.contrast)
    one = jax.vmap(lambda k, im: D._one_view(k, im, dd.img_size, dd.crop_area[0], dd.crop_area[1], rot, flip, br, ct))
    Z = np.zeros((len(X), V, 128), np.float32); T = np.zeros((len(X), V, 9), np.float32); B = np.zeros((len(X), V, 4), np.float32)
    for i0 in range(0, len(X), chunk):
        xb = jnp.asarray(X[i0:i0 + chunk])
        for v in range(V):
            ks = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(key), i0 * 1000 + v), len(xb))
            out, bx, pr = one(ks, xb)
            Z[i0:i0 + chunk, v] = enc(out); T[i0:i0 + chunk, v] = tau_features(bx, pr); B[i0:i0 + chunk, v] = np.asarray(bx)
    return Z, T, B


# ------------------------------------------------------------------ image-space corruption
def corrupt_images(X, kind, seed=0):
    rng = np.random.default_rng(seed); X = X.copy(); n, Hh, Ww, _ = X.shape
    occ = None
    if kind == "noise":
        X = np.clip(X + 0.2 * rng.standard_normal(X.shape).astype(np.float32), 0.0, 1.0)
    elif kind == "occlude":
        s = int(round(Hh * np.sqrt(0.25)))      # 34 px = 24.3% of the image
        r0, c0 = rng.integers(0, Hh - s + 1, n), rng.integers(0, Ww - s + 1, n)
        for i in range(n):
            X[i, r0[i]:r0[i] + s, c0[i]:c0[i] + s] = 0.0
        occ = np.stack([r0, c0, r0 + s, c0 + s], 1)
    return X, occ


def occluded_frac(cy, cx, s, occ, Hh=69):       # fraction of a crop's footprint inside the occluder; cy, cx (n,) or scalar
    lo_r, hi_r = (cy - s + 1) * 0.5 * (Hh - 1), (cy + s + 1) * 0.5 * (Hh - 1)
    lo_c, hi_c = (cx - s + 1) * 0.5 * (Hh - 1), (cx + s + 1) * 0.5 * (Hh - 1)
    ir = np.clip(np.minimum(hi_r, occ[:, 2] - 0.5) - np.maximum(lo_r, occ[:, 0] - 0.5), 0, None)
    ic = np.clip(np.minimum(hi_c, occ[:, 3] - 0.5) - np.maximum(lo_c, occ[:, 1] - 0.5), 0, None)
    return ir * ic / ((hi_r - lo_r) * (hi_c - lo_c))


# ------------------------------------------------------------------ MPC planner
def _bu_one(img_chw, sigma):                    # verbatim port of lgn_glimpser._get_bottom_up_salience
    energy = jnp.sum(jnp.abs(img_chw), axis=0)
    radius = int(sigma * 3)
    if radius % 2 == 0:
        radius += 1
    x = jnp.arange(-radius // 2 + 1, radius // 2 + 1)
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = (kernel / jnp.sum(kernel))[None, None, :, :]
    padded = jnp.pad(energy, ((radius // 2, radius // 2), (radius // 2, radius // 2)), mode="edge")
    dn = jax.lax.ConvDimensionNumbers((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
    return jnp.squeeze(jax.lax.conv_general_dilated(padded[None, None], kernel, (1, 1), "VALID", dimension_numbers=dn))


_bu = jax.jit(jax.vmap(lambda im: _bu_one(jnp.transpose(im, (2, 0, 1)), SIG_BU)))


def bottom_up(X, geom, norm=True):              # S_BU at the geometry's lattice pixels, (n, L)
    m = np.concatenate([np.asarray(_bu(jnp.asarray(X[i:i + 250]))) for i in range(0, len(X), 250)])
    b = m[:, np.rint(geom["py"]).astype(int), np.rint(geom["px"]).astype(int)]
    return b / (b.max(1, keepdims=True) + 1e-12) if norm else b


# Score map at lattice points.  norm=True scales S_TD to max 1 per image (S_BU is scaled in bottom_up): our error
# units differ from MPC's nats, so a raw mix would set beta implicitly.  norm=False is MPC's raw formula (gated).
def plan_scores(bu, hist_idx, hist_err, geom, beta=BETA, norm=True):
    if hist_idx.shape[1] == 0:
        return bu
    py, px = geom["py"], geom["px"]
    d2 = (py[None, :, None] - py[hist_idx][:, None]) ** 2 + (px[None, :, None] - px[hist_idx][:, None]) ** 2
    td = np.sum(hist_err[:, None] * np.exp(-0.5 * d2 / SIG_ERR ** 2), -1)
    if norm:
        td = td / (td.max(1, keepdims=True) + 1e-12)
    ior = np.prod(1.0 - np.exp(-0.5 * d2 / SIG_IOR ** 2), -1)
    return ((1 - beta) * bu + beta * td) * ior


def next_fixation(policy, k, bu, hist_idx, hist_err, geom, rng, perm, occ_score):
    L = geom["L"]
    if policy == "random":
        return perm[:, k]
    if policy == "oracle_occ":                  # least-occluded unvisited point (upper bound for occlusion)
        s = -occ_score.copy(); np.put_along_axis(s, hist_idx, -np.inf, 1) if hist_idx.size else None
        return np.argmax(s, 1)
    if k == 0 or policy in ("epi", "bu"):
        return np.argmax(plan_scores(bu, hist_idx, hist_err, geom, beta=0.0 if policy == "bu" else BETA), 1)
    if policy == "epi_shuf":                    # same planner, errors permuted across images at every step
        sh = np.stack([rng.permutation(hist_err[:, j]) for j in range(hist_err.shape[1])], 1)
        return np.argmax(plan_scores(bu, hist_idx, sh, geom), 1)
    if policy == "ior":                         # farthest-point tour from the same first fixation
        return np.argmax(plan_scores(np.ones_like(bu), hist_idx, np.ones_like(hist_err), geom), 1)
    raise ValueError(policy)


# ------------------------------------------------------------------ predictors
def _cfg_from(d):
    names = {f.name for f in dc.fields(G.GCfg)}
    return G.GCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["cfg"].items() if k in names})


class GPred:                                    # amortised generative PC-ALM: belief = mean amortiser output
    kind = "G"

    def __init__(self, depth, seed, root="pred_runs/tests_sep_fix"):
        base = f"{root}/gpc_sep_d{depth}_amort_hp16_a1_k1_amort_c1_T2200_bp_s{seed}"
        d = json.load(open(base + ".json")); self.cfg = _cfg_from(d); self.seed = seed
        self.p = eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), self.cfg))
        self.am = eqx.tree_deserialise_leaves(base + "_amort.eqx", G.init_amort(jax.random.PRNGKey(0)))
        self.k1_stored = next(r["nmse"] for r in d["rows"] if r["test"] == "k_seen" and r.get("k") == 1)
        H = self.cfg.H
        self._a = eqx.filter_jit(lambda am, x: PF.forward(am, x))
        self._d = eqx.filter_jit(lambda p, s, t: G.topdown(p, s, G.phi(t), H)[1])
        self.predict = G.make_predict(self.p, self.am, self.cfg)

    def embed(self, z, t):                      # (n, m, d), (n, m, 9) -> (n, m, M)  (per-latent contribution)
        n, m = z.shape[:2]; x = np.concatenate([z, t], -1).reshape(n * m, -1)
        return np.concatenate([np.asarray(self._a(self.am, jnp.asarray(x[i:i + 8192]))) for i in range(0, len(x), 8192)]).reshape(n, m, -1)

    def readout(self, s, t, batch=250):         # pooled embedding (n, M), target taus (n, m, 9) -> (n, m, d)
        return np.concatenate([np.asarray(self._d(self.p, jnp.asarray(s[i:i + batch]), jnp.asarray(t[i:i + batch]))) for i in range(0, len(s), batch)])


class FPred:                                    # residual ff PC-ALM: prediction = mean of per-latent predictions
    kind = "F"

    def __init__(self, seed):
        self.rc = RF.RCfg(arm="alm", depth=10, seed=seed, T=400, alpha=1.0, lr=0.01, rho=16.0); self.seed = seed
        self.params = RF.load_model(f"pred_runs/ffres_fix/ffres_almR16_d10_s{seed}.eqx", self.rc)
        self.px = RF.make_predict_x(self.params, self.rc)
        self.predict = RF.make_pair_predict(self.params, self.rc)

    def f(self, z, ta, tb):                     # rows -> rows
        return self.px(np.concatenate([z, ta, tb], -1))


def metrics(P, zte):
    n = len(zte); img = np.repeat(np.arange(n), 4); tv = np.tile(np.arange(4), n)
    m = PE._metrics(P.reshape(-1, P.shape[-1]), zte.reshape(-1, zte.shape[-1]), img, tv, zte)
    return m, np.mean(np.sum((P - zte) ** 2, -1), 1).astype(np.float32)       # per-image squared error over 4 targets


# Pool state: per-kind sums of contributions (G: embeddings; F: predictions of the 4 targets) and counts.
class Pool:
    def __init__(self, model, Zc, Tc, tte):
        self.m, self.Zc, self.Tc, self.tte = model, Zc, Tc, tte
        n = len(Zc); self.n = n
        zp, tp = Zc[:, PERI:PERI + 1], np.repeat(Tc[None, PERI:PERI + 1], n, 0)
        self.cp = self.contrib(zp, tp)[:, 0]
        self.S = {"f": np.zeros_like(self.cp), "p": np.zeros_like(self.cp)}; self.N = {"f": 0, "p": 0}; self.k = 0
        self.seen_z, self.seen_t = [zp], [tp]                                  # for F surprise (peripheral first)

    def contrib(self, z, t):                    # (n, m, d), (n, m, 9) -> G (n, m, M); F (n, m, 4, d)
        if self.m.kind == "G":
            return self.m.embed(z, t)
        n, mm = z.shape[:2]
        rows = [self.m.f(z[:, j], t[:, j], self.tte[:, q]) for j in range(mm) for q in range(4)]
        return np.stack(rows, 1).reshape(n, mm, 4, -1)

    def add(self, z, t, kinds):
        c = self.contrib(z, t)
        for j, kd in enumerate(kinds):
            self.S[kd] = self.S[kd] + c[:, j]; self.N[kd] += 1
        self.k += 1
        self.seen_z.append(z); self.seen_t.append(t)

    def pooled(self, pool):
        if pool == "peri":
            return self.cp
        if pool == "full":
            return (self.S["f"] + self.S["p"] + self.k * self.cp) / (self.N["f"] + self.N["p"] + self.k)
        if pool == "noperi":
            return (self.S["f"] + self.S["p"]) / (self.N["f"] + self.N["p"])
        return self.S[{"fov": "f", "para": "p"}[pool]] / self.N[{"fov": "f", "para": "p"}[pool]]

    def predict(self, pool):
        s = self.pooled(pool)
        return self.m.readout(s, self.tte) if self.m.kind == "G" else s

    # Surprise of the next glimpse's crops under the full pool held BEFORE it (k = 0: the peripheral latent alone).
    def surprise(self, z, t):
        n, mm = z.shape[:2]
        if self.m.kind == "G":
            s = self.pooled("full") if self.k else self.cp
            zh = np.stack([self.m.readout(s, t[:, j:j + 1])[:, 0] for j in range(mm)], 1)
        else:
            zs = np.concatenate(self.seen_z, 1); ts = np.concatenate(self.seen_t, 1); No = zs.shape[1]
            w = np.ones(No); w[0] = max(self.k, 1)                             # peripheral counted once per glimpse
            zh = []
            for j in range(mm):
                pr = self.m.f(zs.reshape(-1, 128), ts.reshape(-1, 9), np.repeat(t[:, j:j + 1], No, 1).reshape(-1, 9)).reshape(n, No, -1)
                zh.append(np.tensordot(pr, w, axes=([1], [0])) / w.sum())
            zh = np.stack(zh, 1)
        return np.mean(np.sum((zh - z) ** 2, -1), 1)


def run_saccades(model, Zc, Tc, tte, zte, X, geom, policy, Kmax, occ, rng, save):
    n, L, cr, kinds = len(zte), geom["L"], geom["crops"], geom["kinds"]
    bu = bottom_up(X, geom)
    perm = rng.permuted(np.tile(np.arange(L), (n, 1)), axis=1)
    occ_score = None
    if occ is not None:                         # mean occluded fraction of each fixation's crops (foveal + parafoveal)
        sp = crop_specs()
        occ_score = np.stack([np.mean([occluded_frac(sp[c][0], sp[c][1], sp[c][2], occ) for c in cr[l]], 0) for l in range(L)], 1)
    pool = Pool(model, Zc, Tc, tte)
    hist_idx = np.zeros((n, 0), np.int64); hist_err = np.zeros((n, 0)); out = {}
    need_err = policy in ("epi", "epi_shuf")
    for k in range(Kmax):
        nxt = next_fixation(policy, k, bu, hist_idx, hist_err, geom, rng, perm, occ_score)
        ids = cr[nxt]                                                           # (n, m)
        z = np.take_along_axis(Zc, ids[..., None], 1); t = Tc[ids]
        err = pool.surprise(z, t) if need_err else np.zeros(n)
        pool.add(z, t, kinds)
        hist_idx = np.concatenate([hist_idx, nxt[:, None]], 1); hist_err = np.concatenate([hist_err, err[:, None]], 1)
        if k + 1 in KS:
            for pl in POOLS:
                m, per = metrics(pool.predict(pl), zte)
                N = {"full": pool.N["f"] + pool.N["p"] + pool.k, "noperi": pool.N["f"] + pool.N["p"], "fov": pool.N["f"], "para": pool.N["p"]}[pl]
                out[(k + 1, pl)] = {**m, "n_latents": N}
                save[(k + 1, pl)] = per
    c = geom["fix"][hist_idx]
    stats = {"mean_fix_radius": float(np.mean(np.hypot(c[..., 0], c[..., 1]))),
             "revisits": float(np.mean([len(r) - len(set(r)) for r in hist_idx])),
             "modal_traj_frac_K8": float(np.mean(np.all(hist_idx[:, :8] == np.array([np.bincount(hist_idx[:, j], minlength=L).argmax() for j in range(min(8, Kmax))]), 1)))}
    if occ is not None:
        sel = np.take_along_axis(occ_score, hist_idx, 1)
        stats["occluded_frac"] = {str(k): float(sel[:, :k].mean()) for k in KS if k <= Kmax}
        stats["occluded_frac_lattice_mean"] = float(occ_score.mean()); stats["occluded_frac_oracle_best"] = float(occ_score.min(1).mean())
    return out, stats, hist_idx.astype(np.int16), hist_err.astype(np.float32)


class BPg:                                      # frozen BP auto-decoder G1: one whole view only
    kind = "BPg"

    def __init__(self, depth, seed):
        base = f"pred_runs/tests/gpc_bp_d{depth}_zero_bp_s{seed}"
        d = json.load(open(base + ".json")); cfg = _cfg_from(d)
        self.k1_stored = next(r["nmse"] for r in d["rows"] if r["test"] == "k_seen" and r.get("k") == 1)
        self.predict = G.make_predict(eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), cfg)), None, cfg)


# Frozen residual BP d10, one whole view only.  Uses the GPU-fix retrain (frozen_bp_fix, job ab3c17c4): the original
# frozen_bp weights score 0.351 only under the faulty GPU numerics they were trained with, 0.393 under correct numerics.
class BPf:
    kind = "BPf"

    def __init__(self, seed):
        rc = RF.RCfg(arm="bp", depth=10, seed=seed, lr=0.001); self.seed = seed
        self.predict = RF.make_pair_predict(RF.load_model(f"pred_runs/frozen_bp_fix/ffres_bp_d10_s{seed}.eqx", rc), rc)
        p = f"pred_runs/tests_ffres_bpfix/ffres_bp_d10_bp_s{seed}.json"
        self.k1_stored = next(r["nmse"] for r in json.load(open(p))["rows"] if r["test"] == "k_seen" and r.get("k") == 1) if os.path.exists(p) else None


def score_whole(model, Zv, Tv, tte, zte, ks, batch=250):
    res = {}
    for k in ks:
        if model.kind in ("G", "BPg"):
            P = np.concatenate([np.asarray(model.predict(Zv[i:i + batch, :k], Tv[i:i + batch, :k], tte[i:i + batch])) for i in range(0, len(zte), batch)])
        else:
            acc = 0
            for v in range(k):
                acc = acc + np.concatenate([np.asarray(model.predict(Zv[i:i + batch, v:v + 1], Tv[i:i + batch, v:v + 1], tte[i:i + batch])) for i in range(0, len(zte), batch)])
            P = acc / k
        res[k] = metrics(P, zte)
    return res


def load_models(seeds, depths=(3, 10)):
    models = {}
    for s in seeds:
        for d in depths:
            models[f"G_d{d}_s{s}"] = GPred(d, s)
            if os.path.exists(f"pred_runs/tests/gpc_bp_d{d}_zero_bp_s{s}.eqx"):      # seed 0 weights were never saved
                models[f"BPg_d{d}_s{s}"] = BPg(d, s)
        models[f"F_d10_s{s}"] = FPred(s); models[f"BPf_d10_s{s}"] = BPf(s)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pred_runs/diag_saccade.json"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=32); ap.add_argument("--vmax", type=int, default=192)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    global KS, KV
    KS = [k for k in KS if k <= a.kmax]; KV = [k for k in KV if k <= a.vmax]
    L_ = np.load("pred_latents.npz")
    raw_tr = L_["bp_emb_tr"]; mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (L_["bp_emb_te"] - mu) / sd, L_["tau_te"].astype(np.float32)
    cfg, params = load_run(BACKBONE); enc = Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32)
    if a.n:
        Xte, zte, tte = Xte[:a.n], zte[:a.n], tte[:a.n]
    models = load_models(a.seeds)
    geoms = {g: make_geom(g) for g in ("grid4", "fov1")}
    res = {"backbone": BACKBONE, "n": len(zte), "kmax": a.kmax, "vmax": a.vmax, "rows": [], "stats": [],
           "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "XLA_FLAGS")},
           "settings": {"A_F": A_F, "A_P": A_P, "h": H_STEP, "sig_bu": SIG_BU, "sig_err": SIG_ERR, "sig_ior": SIG_IOR,
                        "beta": BETA, "occluder_px": 34, "noise_sigma_image": 0.2, "corruption_draws": 1}}
    per_img, traj = {}, {}
    npz = a.out.replace(".json", "_perimage.npz")
    for cond in ("clean", "noise", "occlude"):
        t0 = time.time()
        Xc, occ = corrupt_images(Xte, cond)
        Zc, Tc = crop_table(Xc, enc, cfg.data.img_size)
        views = {v: whole_views(Xc, enc, cfg, max(KV), v == "canon", 303 if v == "aug" else 404) for v in ("aug", "canon")}
        if occ is not None:
            for v, (_, _, B) in views.items():
                fr = np.mean([occluded_frac(B[:, j, 1], B[:, j, 0], B[:, j, 2], occ) for j in range(B.shape[1])])
                res["stats"].append({"cond": cond, "input": f"whole_{v}", "occluded_frac": float(fr)})
        print(cond, "encoded", round(time.time() - t0), "s", flush=True)
        for name, m in models.items():
            for v, (Zv, Tv, _) in views.items():
                ks = [1] if m.kind.startswith("BP") else KV
                if m.kind.startswith("BP") and v == "canon":
                    continue                    # BP: one augmented whole view only (user rule)
                for k, (r, per) in score_whole(m, Zv, Tv, tte, zte, ks).items():
                    res["rows"].append({"model": name, "kind": m.kind, "cond": cond, "input": f"whole_{v}", "geom": "-", "policy": "-", "pool": "-", "K": k, "n_latents": k, **r})
                    per_img[f"{name}|{cond}|whole_{v}|{k}"] = per
            if m.kind not in ("G", "F"):
                continue
            pool = Pool(m, Zc, Tc, tte)
            r, per = metrics(pool.predict("peri"), zte)
            res["rows"].append({"model": name, "kind": m.kind, "cond": cond, "input": "peripheral", "geom": "-", "policy": "-", "pool": "peri", "K": 0, "n_latents": 1, **r})
            per_img[f"{name}|{cond}|peripheral|0"] = per
            for gname, geom in geoms.items():
                pols = POLICIES + (["oracle_occ"] if cond == "occlude" else [])
                for pol in pols:
                    save = {}
                    o, st, hi, he = run_saccades(m, Zc, Tc, tte, zte, Xc, geom, pol, a.kmax, occ, np.random.default_rng(7 + m.seed), save)
                    for (k, pl), rr in o.items():
                        res["rows"].append({"model": name, "kind": m.kind, "cond": cond, "input": "saccade", "geom": gname, "policy": pol, "pool": pl, "K": k, **rr})
                        per_img[f"{name}|{cond}|saccade|{gname}|{pol}|{pl}|{k}"] = save[(k, pl)]
                    traj[f"{name}|{cond}|{gname}|{pol}|idx"] = hi; traj[f"{name}|{cond}|{gname}|{pol}|err"] = he
                    res["stats"].append({"model": name, "cond": cond, "geom": gname, "policy": pol, **st})
                    print(cond, name, gname, pol, {k: round(rr["nmse"], 4) for (k, pl), rr in o.items() if pl == "full"}, round(time.time() - t0), "s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        np.savez_compressed(npz, **per_img, **traj, zte_sqnorm=np.sum(zte ** 2, -1).astype(np.float32))
    res["per_image_file"] = os.path.basename(npz)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
