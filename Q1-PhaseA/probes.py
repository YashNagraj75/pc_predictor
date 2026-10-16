import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


# ===================================================== 1. scene quantities
# imgs: (N,H,W,C) float in [0,1], must be same pixels used for embeddings.
# Returns {name: (N,) float64}. Flux-weighted image moments + 2 morphology summaries.
# Targets computed by fixed deterministic rule (no labels needed), unlike LeWM's simulator truth.
# Orientation = sin/cos of 2x position angle (PA defined mod pi; avoids wrap-around penalty).
# Measures what embedding retains, not invariance -- see augmentation_factors for that.
def image_factors(imgs, eps=1e-8):
    X = np.asarray(imgs, np.float64)
    if X.ndim != 4:
        raise ValueError(f"image_factors needs (N,H,W,C), got {X.shape}")
    N, H, W, C = X.shape
    I = X.mean(-1)                                    # (N,H,W) luminance

    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W),
                         indexing="ij")
    tot = I.sum((1, 2)) + eps
    cx = (I * xx).sum((1, 2)) / tot
    cy = (I * yy).sum((1, 2)) / tot

    dx = xx[None] - cx[:, None, None]
    dy = yy[None] - cy[:, None, None]
    Mxx = (I * dx * dx).sum((1, 2)) / tot
    Myy = (I * dy * dy).sum((1, 2)) / tot
    Mxy = (I * dx * dy).sum((1, 2)) / tot

    tr = Mxx + Myy + eps
    r_rms = np.sqrt(np.maximum(tr, 0.0))              # flux-weighted size
    e1 = (Mxx - Myy) / tr                             # ellipticity components
    e2 = 2.0 * Mxy / tr
    ellip = np.sqrt(e1 ** 2 + e2 ** 2)
    pa2 = np.arctan2(e2, e1)                          # 2 x position angle

    # concentration: flux inside half the rms radius / total flux
    r = np.sqrt(dx ** 2 + dy ** 2)
    conc = (I * (r < 0.5 * r_rms[:, None, None])).sum((1, 2)) / tot

    # asymmetry: residual under 180-degree rotation about the image centre
    asym = np.abs(I - I[:, ::-1, ::-1]).sum((1, 2)) / (2.0 * tot)

    out = {
        "flux":          tot,
        "centroid_x":    cx,
        "centroid_y":    cy,
        "size_rms":      r_rms,
        "ellipticity":   ellip,
        "orient_sin2pa": np.sin(pa2),
        "orient_cos2pa": np.cos(pa2),
        "concentration": conc,
        "asymmetry":     asym,
        "contrast_std":  I.std((1, 2)),
    }
    if C == 3:
        # colour is physical for galaxies (red ellipticals vs blue spirals)
        out["colour_rg"] = X[..., 0].mean((1, 2)) - X[..., 1].mean((1, 2))
        out["colour_gb"] = X[..., 1].mean((1, 2)) - X[..., 2].mean((1, 2))
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


# Probe targets for quantities DELIBERATELY augmented away.
# boxes: (B,4) crop boxes (cx,cy,w,h) from data.make_views
# params: (B,4) optional (theta, flip, brightness, contrast)
# Inverted vs image_factors: low r is GOOD (invariance held).
# Kept separate from image_factors since goodness direction is opposite.
def augmentation_factors(boxes, params=None):
    b = np.asarray(boxes, np.float64)
    out = {"aug_crop_cx": b[:, 0], "aug_crop_cy": b[:, 1], "aug_crop_scale": b[:, 2]}
    if params is not None:
        p = np.asarray(params, np.float64)
        out["aug_rot_sin"] = np.sin(p[:, 0])
        out["aug_rot_cos"] = np.cos(p[:, 0])
        out["aug_flip"] = p[:, 1]
        out["aug_brightness"] = p[:, 2]
        out["aug_contrast"] = p[:, 3]
    return out


# ============================================================== 2. probes
def _standardise(Ztr, Zte):
    mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-8
    return (Ztr - mu) / sd, (Zte - mu) / sd


# Closed-form ridge regression probe. -> {nmse, mse, r, alpha}
# Ridge not OLS: D can approach N on hidden layers, unregularized fit
# would report interpolation threshold rather than decodability.
# alpha chosen on held-out slice of TRAIN -- never on test.
def ridge_probe(Ztr, ttr, Zte, tte, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3),
                val_frac=0.2, seed=0):
    Ztr, Zte = np.asarray(Ztr, np.float64), np.asarray(Zte, np.float64)
    ttr, tte = np.asarray(ttr, np.float64).ravel(), np.asarray(tte, np.float64).ravel()
    Ztr, Zte = _standardise(Ztr, Zte)
    tmu, tsd = ttr.mean(), ttr.std() + 1e-12

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Ztr))
    nval = max(1, int(val_frac * len(Ztr)))
    vi, ti = perm[:nval], perm[nval:]

    def fit(Z, t, a):
        Z1 = np.concatenate([Z, np.ones((len(Z), 1))], 1)
        A = Z1.T @ Z1
        A[np.arange(Z.shape[1]), np.arange(Z.shape[1])] += a   # no penalty on bias
        return np.linalg.solve(A, Z1.T @ t)

    best, best_a = np.inf, alphas[0]
    for a in alphas:
        w = fit(Ztr[ti], ttr[ti], a)
        pv = np.concatenate([Ztr[vi], np.ones((nval, 1))], 1) @ w
        e = float(((pv - ttr[vi]) ** 2).mean())
        if e < best:
            best, best_a = e, a

    w = fit(Ztr, ttr, best_a)
    pred = np.concatenate([Zte, np.ones((len(Zte), 1))], 1) @ w
    return {**_reg_scores(pred, tte, tsd), "alpha": float(best_a)}


def _reg_scores(pred, targ, tsd):
    mse = float(((pred - targ) ** 2).mean())
    if pred.std() < 1e-12 or targ.std() < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(pred, targ)[0, 1])
    return {"nmse": float(mse / (tsd ** 2)), "mse": mse, "r": r}


# Two-layer MLP probe, early-stopped. -> {nmse, mse, r, best_step}
# vs ridge_probe: distinguishes 'discarded' from 'kept but entangled nonlinearly'
# EARLY STOPPING IS LOAD-BEARING: without it, high-capacity probe overfits
# and can score WORSE than closed-form ridge (incoherent as a measurement).
# Holds out slice of TRAIN for best-val params; TEST touched exactly once at end.
def mlp_probe(Ztr, ttr, Zte, tte, hidden=256, steps=1500,
              lrs=(3e-4, 1e-3, 3e-3), batch=512, seed=0, val_frac=0.2,
              eval_every=50,
              weight_decay=1e-4):
    Ztr, Zte = _standardise(np.asarray(Ztr, np.float32), np.asarray(Zte, np.float32))
    ttr = np.asarray(ttr, np.float64).ravel()
    tte = np.asarray(tte, np.float64).ravel()
    tmu, tsd = ttr.mean(), ttr.std() + 1e-12
    y_all = ((ttr - tmu) / tsd).astype(np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Ztr))
    nval = max(1, int(val_frac * len(Ztr)))
    vi, fi = perm[:nval], perm[nval:]

    Zf, yf = jnp.asarray(Ztr[fi]), jnp.asarray(y_all[fi])
    Zv, yv = jnp.asarray(Ztr[vi]), jnp.asarray(y_all[vi])

    def loss_fn(m, zb, yb):
        return jnp.mean((jax.vmap(m)(zb).squeeze(-1) - yb) ** 2)

    @eqx.filter_jit
    def val_loss(m):
        return loss_fn(m, Zv, yv)

    n = len(Zf)
    best, best_net, best_step, best_lr = float("inf"), None, 0, None
    for lr in lrs:
        key = jax.random.PRNGKey(seed)
        k1, kb = jax.random.split(key)
        net = eqx.nn.MLP(Ztr.shape[1], 1, hidden, 1,
                         activation=jax.nn.gelu, key=k1)
        opt = optax.adamw(lr, weight_decay=weight_decay)
        state = opt.init(eqx.filter(net, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(m, st, zb, yb):
            l, g = eqx.filter_value_and_grad(loss_fn)(m, zb, yb)
            upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
            return eqx.apply_updates(m, upd), st, l

        for i in range(steps):
            kb, sub = jax.random.split(kb)
            idx = jax.random.choice(sub, n, (min(batch, n),), replace=False)
            net, state, _ = step(net, state, Zf[idx], yf[idx])
            if (i + 1) % eval_every == 0:
                v = float(val_loss(net))
                if v < best:
                    best, best_net, best_step, best_lr = v, net, i + 1, lr

    pred = np.asarray(jax.vmap(best_net)(jnp.asarray(Zte)).squeeze(-1), np.float64)
    pred = pred * tsd + tmu
    out = _reg_scores(pred, tte, tsd)
    out["best_step"] = int(best_step)
    out["best_lr"] = float(best_lr)
    return out


# Linear (hidden=None) or MLP softmax probe, early-stopped over small LR grid.
# Returns {accuracy, chance, best_step, best_lr, val_accuracy}; LR selected on val, not fixed.
# Complements knn_accuracy (capacity-free, local): this finds globally-linear structure.
# Model selection uses held-out slice of TRAIN (see mlp_probe) to avoid hidden=256 overfitting.
# Both variants selected identically so linear-vs-MLP gap stays interpretable.
def class_probe(Ztr, ytr, Zte, yte, hidden=None, steps=2000,
                lrs=(3e-4, 1e-3, 3e-3), batch=512, seed=0, val_frac=0.2,
                eval_every=50, weight_decay=1e-4):
    Ztr, Zte = _standardise(np.asarray(Ztr, np.float32), np.asarray(Zte, np.float32))
    ytr = np.asarray(ytr).ravel().astype(np.int32)
    yte = np.asarray(yte).ravel().astype(np.int32)
    n_cls = int(max(ytr.max(), yte.max())) + 1

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Ztr))
    nval = max(1, int(val_frac * len(Ztr)))
    vi, fi = perm[:nval], perm[nval:]

    Zf, yf = jnp.asarray(Ztr[fi]), jnp.asarray(ytr[fi])
    Zv, yv = jnp.asarray(Ztr[vi]), jnp.asarray(ytr[vi])

    def loss_fn(m, zb, yb):
        lg = jax.vmap(m)(zb)
        return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(len(yb)), yb])

    @eqx.filter_jit
    def val_acc(m):
        return jnp.mean(jax.vmap(m)(Zv).argmax(-1) == yv)

    n = len(Zf)
    best, best_net, best_step, best_lr = -1.0, None, 0, None
    for lr in lrs:
        key = jax.random.PRNGKey(seed)
        k1, kb = jax.random.split(key)
        if hidden is None:
            net = eqx.nn.Linear(Ztr.shape[1], n_cls, key=k1)
        else:
            net = eqx.nn.MLP(Ztr.shape[1], n_cls, hidden, 1,
                             activation=jax.nn.gelu, key=k1)
        opt = optax.adamw(lr, weight_decay=weight_decay)
        state = opt.init(eqx.filter(net, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(m, st, zb, yb):
            l, g = eqx.filter_value_and_grad(loss_fn)(m, zb, yb)
            upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
            return eqx.apply_updates(m, upd), st, l

        for i in range(steps):
            kb, sub = jax.random.split(kb)
            idx = jax.random.choice(sub, n, (min(batch, n),), replace=False)
            net, state, _ = step(net, state, Zf[idx], yf[idx])
            if (i + 1) % eval_every == 0:
                a = float(val_acc(net))
                if a > best:
                    best, best_net, best_step, best_lr = a, net, i + 1, lr

    pred = np.asarray(jax.vmap(best_net)(jnp.asarray(Zte)).argmax(-1))
    return {"accuracy": float((pred == yte).mean()),
            "chance": float(1.0 / n_cls),
            "val_accuracy": float(best),
            "best_step": int(best_step),
            "best_lr": float(best_lr)}


# LeWM Table-1 analogue: every target x {linear, MLP}.
# targets_* : {name: (N,)} from image_factors / augmentation_factors.
# Returns {"regression": {name: {"linear": {...}, "mlp": {...}}},
# "classification": {"linear": {...}, "mlp": {...}}}
def probe_report(Ztr, Zte, targets_tr, targets_te, ytr=None, yte=None,
                 mlp=True, seed=0):
    out = {"regression": {}, "classification": {}}
    for name in sorted(targets_tr):
        row = {"linear": ridge_probe(Ztr, targets_tr[name], Zte,
                                     targets_te[name], seed=seed)}
        if mlp:
            row["mlp"] = mlp_probe(Ztr, targets_tr[name], Zte,
                                   targets_te[name], seed=seed)
        out["regression"][name] = row
    if ytr is not None:
        out["classification"]["linear"] = class_probe(Ztr, ytr, Zte, yte,
                                                      hidden=None, seed=seed)
        if mlp:
            out["classification"]["mlp"] = class_probe(Ztr, ytr, Zte, yte,
                                                       hidden=256, seed=seed)
    return out


# ============================================================= 3. decoder
# Shared decoder: embedding -> pixels. Fixed architecture on purpose, pinned not configurable.
# Pins arch so SSIM is comparable across all runs; changing this class breaks comparability -- version instead of editing.
# RECONSTRUCTION IS NEVER PART OF TRAINING: encoder frozen, only decoder learns.
# Measures what the embedding retained, not what encoder could be made to retain.
class Decoder(eqx.Module):
    fc: eqx.nn.Linear
    convs: list
    head: eqx.nn.Conv2d
    base: int
    ch: int

    def __init__(self, embed_dim, out_hw, out_ch, width=128, base=4, *, key):
        ks = jax.random.split(key, 5)
        self.base, self.ch = base, out_ch
        self.fc = eqx.nn.Linear(embed_dim, width * base * base, key=ks[0])
        n_up = int(np.log2(out_hw // base))
        chans = [width // (2 ** i) for i in range(n_up + 1)]
        self.convs = [eqx.nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, key=ks[1 + i])
                      for i in range(n_up)]
        self.head = eqx.nn.Conv2d(chans[n_up], out_ch, 3, padding=1, key=ks[-1])

    def __call__(self, z):
        h = self.fc(z).reshape(-1, self.base, self.base)
        for cv in self.convs:
            h = jnp.repeat(jnp.repeat(h, 2, axis=1), 2, axis=2)   # nearest-neighbour up
            h = jax.nn.gelu(cv(h))
        return jax.nn.sigmoid(self.head(h))                        # (C,H,W) in [0,1]


# Fit Decoder on FROZEN embeddings -> returns decoder
# Ztr: (N, embed_dim), Xtr: (N, H, W, C) in [0,1]
def train_decoder(Ztr, Xtr, embed_dim, img_shape, steps=3000, lr=1e-3,
                  batch=128, width=128, seed=0, verbose=False):
    H, W, C = img_shape
    assert H == W, "square images assumed"
    key = jax.random.PRNGKey(seed)
    k1, kb = jax.random.split(key)
    dec = Decoder(embed_dim, H, C, width=width, key=k1)
    opt = optax.adam(lr)
    state = opt.init(eqx.filter(dec, eqx.is_inexact_array))

    Zj = jnp.asarray(np.asarray(Ztr, np.float32))
    # store as (N,C,H,W) to match conv layout
    Xj = jnp.asarray(np.transpose(np.asarray(Xtr, np.float32), (0, 3, 1, 2)))

    def loss_fn(m, zb, xb):
        return jnp.mean((jax.vmap(m)(zb) - xb) ** 2)

    @eqx.filter_jit
    def step(m, st, zb, xb):
        l, g = eqx.filter_value_and_grad(loss_fn)(m, zb, xb)
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, l

    n = len(Zj)
    for i in range(steps):
        kb, sub = jax.random.split(kb)
        idx = jax.random.choice(sub, n, (min(batch, n),), replace=False)
        dec, state, l = step(dec, state, Zj[idx], Xj[idx])
        if verbose and i % 500 == 0:
            print(f"  decoder step {i:5d}  mse={float(l):.5f}", flush=True)
    return dec


def _gauss_win(size=11, sigma=1.5):
    g = np.exp(-((np.arange(size) - size // 2) ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


# Mean SSIM over a batch. A, B : (N,H,W,C) in [0, data_range].
# Written out rather than imported so suite has no scikit-image dependency
# and window/constants are visible and fixed.
# SSIM numbers only comparable at identical settings; MPC Table 1 omits them.
def ssim(A, B, data_range=1.0, size=11, sigma=1.5):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    w = _gauss_win(size, sigma)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    pad = size // 2

    def filt(X):                     # (N,H,W) valid-region correlation
        N, H, W_ = X.shape
        out = np.empty((N, H - 2 * pad, W_ - 2 * pad))
        for i in range(size):
            for j in range(size):
                sl = X[:, i:i + H - 2 * pad, j:j + W_ - 2 * pad]
                if i == 0 and j == 0:
                    out[:] = w[i, j] * sl
                else:
                    out += w[i, j] * sl
        return out

    vals = []
    for c in range(A.shape[-1]):
        a, b = A[..., c], B[..., c]
        ma, mb = filt(a), filt(b)
        saa = filt(a * a) - ma * ma
        sbb = filt(b * b) - mb * mb
        sab = filt(a * b) - ma * mb
        num = (2 * ma * mb + C1) * (2 * sab + C2)
        den = (ma ** 2 + mb ** 2 + C1) * (saa + sbb + C2)
        vals.append(num / np.maximum(den, 1e-12))
    return float(np.mean(vals))


# Returns {mse, psnr, ssim} on held-out images, plus the reconstructions.
def decode_report(dec, Zte, Xte):
    recon = np.asarray(jax.vmap(dec)(jnp.asarray(np.asarray(Zte, np.float32))))
    recon = np.transpose(recon, (0, 2, 3, 1))          # -> (N,H,W,C)
    X = np.asarray(Xte, np.float64)
    mse = float(((recon - X) ** 2).mean())
    return {"mse": mse,
            "psnr": float(10 * np.log10(1.0 / max(mse, 1e-12))),
            "ssim": ssim(X, recon)}, recon


# Floor for decode_report: best possible CONSTANT reconstruction.
# Decoder ignoring embedding, emitting dataset mean image, scores decent SSIM
# on centred, low-diversity datasets -- SSIM alone is unreadable otherwise.
# Any claim embedding "retains the scene" must beat this row.
def baseline_decode_report(Xtr, Xte):
    mean_img = np.asarray(Xtr, np.float64).mean(0, keepdims=True)
    X = np.asarray(Xte, np.float64)
    rec = np.repeat(mean_img, len(X), 0)
    mse = float(((rec - X) ** 2).mean())
    return {"mse": mse,
            "psnr": float(10 * np.log10(1.0 / max(mse, 1e-12))),
            "ssim": ssim(X, rec)}


# ================================================== 4. path straightness
# LeWM Eq. 9 analogue. Zseq: (T, B, D) -> {mean, per_step}
# Mean cosine sim between consecutive velocity vectors v_t = z_{t+1}-z_t.
# 1.0 = perfectly straight path, 0 = directionless.
# Phase A: only PC relaxation trace use is available (inference, not representation).
# Do not report this as the actual LeWM measurement (needs glimpse/saccade sequence).
def path_straightness(Zseq):
    Z = np.asarray(Zseq, np.float64)
    if Z.ndim != 3 or Z.shape[0] < 3:
        raise ValueError(f"path_straightness needs (T>=3, B, D), got {Z.shape}")
    v = np.diff(Z, axis=0)
    nv = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    cos = (nv[:-1] * nv[1:]).sum(-1)                   # (T-2, B)
    return {"mean": float(cos.mean()),
            "per_step": [float(c) for c in cos.mean(1)]}


# ====================================================== 5. visualisation
# 2-D t-SNE.
# Returns: coords (n,2), index into Z
def tsne_embedding(Z, seed=0, perplexity=30, max_n=3000):
    from sklearn.manifold import TSNE
    Z = np.asarray(Z, np.float64)
    idx = np.arange(len(Z))
    if len(Z) > max_n:
        idx = np.random.default_rng(seed).choice(len(Z), max_n, replace=False)
    ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
              random_state=seed)
    return ts.fit_transform(Z[idx]), idx


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "figure.facecolor": "white",
    })


# t-SNE scatter: one panel colored by class, optionally one by continuous factor.
# Ticks dropped -- t-SNE coordinates have no units.
def tsne_figure(Z, y, path, colour_by=None, colour_name="", title="",
                seed=0, class_names=None):
    import matplotlib.pyplot as plt
    _style()
    XY, idx = tsne_embedding(Z, seed=seed)
    ncol = 1 if colour_by is None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.5), squeeze=False)

    ax = axes[0, 0]
    yy = np.asarray(y).ravel()[idx]
    for c in np.unique(yy):
        m = yy == c
        lab = class_names[int(c)] if class_names else f"{int(c)}"
        ax.scatter(XY[m, 0], XY[m, 1], s=3, alpha=0.6, linewidths=0, label=lab)
    ax.set_title("coloured by class")
    ax.legend(frameon=False, markerscale=3, ncol=2, loc="best",
              handletextpad=0.2, columnspacing=0.6)

    if colour_by is not None:
        ax2 = axes[0, 1]
        cv = np.asarray(colour_by).ravel()[idx]
        sc = ax2.scatter(XY[:, 0], XY[:, 1], c=cv, s=3, alpha=0.75,
                         linewidths=0, cmap="viridis")
        ax2.set_title(f"coloured by {colour_name}")
        cb = fig.colorbar(sc, ax=ax2, fraction=0.045, pad=0.02)
        cb.set_label(colour_name, fontsize=7)

    for ax_ in axes.ravel():
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.margins(0.04)
    if title:
        fig.suptitle(title, fontsize=9, x=0.02, ha="left")
    fig.savefig(path)
    plt.close(fig)
    return path


# Originals / reconstructions / (optional) constant-mean floor.
# Row 3 is the point: compare row2 vs row1 is meaningless without seeing
# what a trivial decoder (ignoring embedding) produces as baseline.
def reconstruction_figure(X, recon, path, n=8, title="", mean_img=None):
    import matplotlib.pyplot as plt
    _style()
    n = min(n, len(X))
    rows = 2 if mean_img is None else 3
    fig, axes = plt.subplots(rows, n, figsize=(1.05 * n, 1.15 * rows),
                             squeeze=False)
    labels = ["input", "decoded", "mean-image\nfloor"][:rows]
    for j in range(n):
        ims = [X[j], recon[j]] + ([mean_img[0]] if mean_img is not None else [])
        for i, im in enumerate(ims):
            a = axes[i, j]
            a.imshow(np.clip(im.squeeze(), 0, 1),
                     cmap=None if im.shape[-1] == 3 else "gray",
                     vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if j == 0:
                a.set_ylabel(labels[i], fontsize=7, rotation=0, ha="right",
                             va="center", labelpad=6)
    if title:
        fig.suptitle(title, fontsize=9, x=0.02, ha="left")
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================ 6. run_suite
# Runs everything in one call -> dict (json-serialisable), written to disk.
# Ztr/Zte: (N,D) embeddings from FROZEN encoder; Xtr/Xte: (N,H,W,C) images in [0,1]; ytr/yte: (N,) labels
# Writes <out_dir>/probes<tag>.json plus two figures.
# Returns the same dict so a sweep can collect results without re-reading disk.
def run_suite(Ztr, Zte, Xtr, Xte, ytr, yte, out_dir, *, tag="",
              boxes_te=None, aug_params_te=None, boxes_tr=None,
              aug_params_tr=None, Zaug_tr=None, Zaug_te=None,
              decoder_steps=3000, probe_mlp=True,
              make_figures=True, seed=0, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    Xtr = np.asarray(Xtr, np.float32); Xte = np.asarray(Xte, np.float32)
    img_shape = Xtr.shape[1:]
    rep = {"tag": tag, "n_train": int(len(Ztr)), "n_test": int(len(Zte)),
           "embed_dim": int(np.shape(Ztr)[1]), "img_shape": list(img_shape)}

    if verbose:
        print("[probes] scene-quantity probes ...", flush=True)
    Ftr, Fte = image_factors(Xtr), image_factors(Xte)
    rep["scene"] = probe_report(Ztr, Zte, Ftr, Fte, ytr, yte,
                                mlp=probe_mlp, seed=seed)

    if boxes_te is not None and boxes_tr is not None:
        if verbose:
            print("[probes] augmentation-factor probes ...", flush=True)
        if Zaug_tr is None or Zaug_te is None:
            raise ValueError(
                "augmentation probes require Zaug_tr/Zaug_te -- embeddings of "
                "the AUGMENTED views that boxes_*/aug_params_* describe. "
                "Handing them the deterministic centre-view embeddings "
                "(Ztr/Zte) asks whether a crop box is recoverable from an "
                "image that was never cropped, which is unanswerable by "
                "construction and would score as a spurious invariance.")
        Atr = augmentation_factors(boxes_tr, aug_params_tr)
        Ate = augmentation_factors(boxes_te, aug_params_te)
        rep["augmentation"] = probe_report(Zaug_tr, Zaug_te, Atr, Ate,
                                           mlp=probe_mlp, seed=seed)
        rep["augmentation_note"] = (
            "Read INVERTED relative to 'scene': the objective asks the encoder "
            "to be invariant to these, so LOW r / HIGH nmse is the good "
            "outcome. Recovering a factor means the invariance did not take.")

    if verbose:
        print("[probes] decoder ...", flush=True)
    dec = train_decoder(Ztr, Xtr, rep["embed_dim"], img_shape,
                        steps=decoder_steps, seed=seed, verbose=verbose)
    rep["decode"], recon = decode_report(dec, Zte, Xte)
    rep["decode_floor"] = baseline_decode_report(Xtr, Xte)
    rep["decode"]["ssim_gain_over_floor"] = (
        rep["decode"]["ssim"] - rep["decode_floor"]["ssim"])

    if make_figures:
        if verbose:
            print("[probes] figures ...", flush=True)
        # Figures are a convenience; the probe and decoder numbers above are
        # the deliverable.  A missing optional plotting dependency must not
        # discard a completed run, so each figure is attempted independently
        # and its failure recorded under rep["fig_errors"].
        rep["fig_errors"] = {}
        mean_img = Xtr.mean(0, keepdims=True)
        try:
            rep["fig_recon"] = reconstruction_figure(
                Xte, recon, os.path.join(out_dir, f"probes_recon{tag}.png"),
                title=f"Decoded from frozen embedding  {tag}", mean_img=mean_img)
        except Exception as e:
            rep["fig_errors"]["recon"] = f"{type(e).__name__}: {e}"
            if verbose:
                print("[probes] recon figure skipped:", e, flush=True)
        try:
            rep["fig_tsne"] = tsne_figure(
                Zte, yte, os.path.join(out_dir, f"probes_tsne{tag}.png"),
                colour_by=Fte["size_rms"], colour_name="size (rms radius)",
                title=f"Latent structure  {tag}", seed=seed)
        except Exception as e:
            rep["fig_errors"]["tsne"] = f"{type(e).__name__}: {e}"
            if verbose:
                print("[probes] t-SNE figure skipped:", e, flush=True)

    with open(os.path.join(out_dir, f"probes{tag}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep


# One flat row per run, for cross-run tables. -> dict
def summarise(rep, targets=("flux", "size_rms", "ellipticity",
                            "concentration", "colour_rg")):
    row = {"tag": rep.get("tag", "")}
    for t in targets:
        r = rep.get("scene", {}).get("regression", {}).get(t)
        if r:
            row[f"{t}_lin_r"] = r["linear"]["r"]
            row[f"{t}_lin_nmse"] = r["linear"]["nmse"]
            if "mlp" in r:
                row[f"{t}_mlp_r"] = r["mlp"]["r"]
    cl = rep.get("scene", {}).get("classification", {})
    if "linear" in cl:
        row["class_lin_acc"] = cl["linear"]["accuracy"]
    if "mlp" in cl:
        row["class_mlp_acc"] = cl["mlp"]["accuracy"]
    if "decode" in rep:
        row["decode_ssim"] = rep["decode"]["ssim"]
        row["decode_ssim_floor"] = rep["decode_floor"]["ssim"]
        row["decode_ssim_gain"] = rep["decode"]["ssim_gain_over_floor"]
        row["decode_psnr"] = rep["decode"]["psnr"]
    return row
