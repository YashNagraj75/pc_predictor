"""Shared, pre-registered evaluation for the Phase P predictors (SDPC, BP set, feedforward, later generative PC).

A predictor is a callable   predict(Zs, Ts, Tt) -> Zhat
    Zs (n, k, d)  standardised latents of the k SEEN views     Ts (n, k, 9) their transforms
    Tt (n, m, 9)  transforms of the m TARGET views             Zhat (n, m, d)

Cases (fixed, seeded, identical for every method):
    k_cases(k): for each test image a permutation of its 4 views from rng(1000 + k); the first k
                are seen, the remaining 4 - k are targets.
    pair_cases(): all 12 ordered pairs (A -> B), exactly the feedforward track's pairs.
Metrics exactly as pred_ff.evaluate: nmse = E||zhat - z||^2 / E||z||^2 (standardised latents),
mean cosine, retrieval@1 = zhat's nearest neighbour among the 2000 test images' latents of the
SAME target view index is the right image.

Corruption (robustness test): the stored test latents came from pred_data.py (key 202, chunks of
512 folded in, data.make_views_labeled).  test_views() reproduces those pixels exactly and
encode() re-encodes them; callers must check clean re-encodings match the stored latents
(check_reproduction) before trusting corrupted ones.  One SEEN view per image is corrupted:
Gaussian noise (sigma on [0,1] pixels, clipped) or a 16x16 (25 % area) occlusion square at a
seeded position, filled with 0.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp

N_VIEWS = 4


def k_cases(n, k):
    rng = np.random.default_rng(1000 + k)
    perm = np.stack([rng.permutation(N_VIEWS) for _ in range(n)])
    return perm[:, :k], perm[:, k:]


def pair_cases(n):
    pairs = [(a, b) for a in range(N_VIEWS) for b in range(N_VIEWS) if a != b]
    return [(np.full((n, 1), a), np.full((n, 1), b)) for a, b in pairs]


def _metrics(P, Z, img, tview, zte_all, n_retr=2000):
    nmse = float(np.mean(np.sum((P - Z) ** 2, -1)) / np.mean(np.sum(Z ** 2, -1)))
    cos = float(np.mean(np.sum(P * Z, -1) / (np.linalg.norm(P, axis=-1) * np.linalg.norm(Z, axis=-1) + 1e-12)))
    hits = []
    keep = img < n_retr
    for b in range(N_VIEWS):
        sel = keep & (tview == b)
        if not sel.any():
            continue
        C = zte_all[:n_retr, b]
        d2 = np.sum(P[sel] ** 2, -1)[:, None] - 2 * P[sel] @ C.T + np.sum(C ** 2, -1)[None]
        hits.append(np.argmin(d2, 1) == img[sel])
    return {"nmse": nmse, "cos": cos, "retr1": float(np.mean(np.concatenate(hits)))}


def run_cases(predict, zte, tte, seen, tgt, zseen=None, batch=500):
    """zseen: optional replacement latents for the seen views (n, 4, d) -- e.g. corrupted re-encodings."""
    n = zte.shape[0]
    src = zte if zseen is None else zseen
    ii = np.arange(n)[:, None]
    P = []
    for i in range(0, n, batch):
        s = slice(i, i + batch)
        P.append(np.asarray(predict(src[ii[s], seen[s]], tte[ii[s], seen[s]], tte[ii[s], tgt[s]])))
    P = np.concatenate(P)                                          # (n, m, d)
    Z = zte[ii, tgt]
    img = np.repeat(np.arange(n), tgt.shape[1]); tview = tgt.reshape(-1)
    return _metrics(P.reshape(-1, P.shape[-1]), Z.reshape(-1, Z.shape[-1]), img, tview, zte)


def eval_pairs(predict, zte, tte, zseen=None):
    """The 12 ordered pairs, pooled -- comparable to pred_ff.evaluate."""
    n = zte.shape[0]
    Ps, Zs, imgs, tv = [], [], [], []
    ii = np.arange(n)[:, None]
    src = zte if zseen is None else zseen
    for seen, tgt in pair_cases(n):
        P = np.concatenate([np.asarray(predict(src[ii[i:i + 500], seen[i:i + 500]], tte[ii[i:i + 500], seen[i:i + 500]],
                                               tte[ii[i:i + 500], tgt[i:i + 500]])) for i in range(0, n, 500)])
        Ps.append(P[:, 0]); Zs.append(zte[np.arange(n), tgt[:, 0]]); imgs.append(np.arange(n)); tv.append(tgt[:, 0])
    return _metrics(np.concatenate(Ps), np.concatenate(Zs), np.concatenate(imgs), np.concatenate(tv), zte)


# ------------------------------------------------------------------ pixels and corruption
def test_views(cfg_data, Xte, chunk=512, key=202):
    import data as D
    out = []
    for i in range(0, len(Xte), chunk):
        k = jax.random.fold_in(jax.random.PRNGKey(key), i)
        Zin, _, _ = D.make_views_labeled(k, jnp.asarray(Xte[i:i + chunk]), cfg_data)
        out.append(np.asarray(Zin).transpose(1, 0, 2))
    return np.concatenate(out)                                     # (n, 4, 3072)


def encode(params, cfg_enc, pix, batch=1024):
    import equinox as eqx, encoder as E
    f = eqx.filter_jit(lambda p, x: E.forward(p, x, cfg_enc)[0])
    flat = pix.reshape(-1, pix.shape[-1])
    z = np.concatenate([np.asarray(f(params, jnp.asarray(flat[i:i + batch]))) for i in range(0, len(flat), batch)])
    return z.reshape(pix.shape[0], pix.shape[1], -1)


def corrupt(pix, which, kind, level, img_size=32, seed=0):
    """Corrupt view which[i] of image i.  kind 'noise' (level = sigma) or 'occlude' (level = area fraction)."""
    rng = np.random.default_rng(seed)
    out = pix.copy()
    n = pix.shape[0]
    v = out[np.arange(n), which].reshape(n, img_size, img_size, -1)
    if kind == "noise":
        v = np.clip(v + level * rng.standard_normal(v.shape).astype(np.float32), 0.0, 1.0)
    elif kind == "occlude":
        s = int(round(img_size * np.sqrt(level)))
        r0, c0 = rng.integers(0, img_size - s + 1, n), rng.integers(0, img_size - s + 1, n)
        for i in range(n):
            v[i, r0[i]:r0[i] + s, c0[i]:c0[i] + s] = 0.0
    else:
        raise ValueError(kind)
    out[np.arange(n), which] = v.reshape(n, -1)
    return out


def check_reproduction(z_reenc, z_stored, tol=1e-3):
    err = float(np.max(np.abs(z_reenc - z_stored)) / (np.max(np.abs(z_stored)) + 1e-12))
    assert err < tol, f"re-encoded test views do not reproduce the stored latents (rel max err {err:.2e})"
    return err
