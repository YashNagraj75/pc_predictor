# Representation-quality metrics shared by the Q1 probes: linear CKA and KS two-sample tests between two encoders'
# representations, spectral diagnostics (eigenspectrum, effective rank, intrinsic dimension), k-NN accuracy, and
# cheap collapse checks. Spectral metrics are only valid on hidden layers -- SIGReg pins the output embedding to N(0,I).

import numpy as np


def linear_cka(X, Y):
    """Computes linear CKA between two representations X (N, D1) and Y (N, D2) of matched inputs.
    Returns a scalar in [0, 1]; a low value means the reps differ, not that one is worse."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    assert X.shape[0] == Y.shape[0], "CKA needs matched rows"
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = X.T @ Y
    num = (xty ** 2).sum()
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else 0.0


def eig_spectrum(X):
    """Computes covariance eigenvalues of X in descending order via SVD of the centered matrix.
    Uses SVD instead of forming X^T X for numerical stability."""
    X = np.asarray(X, np.float64)
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return np.maximum(s ** 2 / max(X.shape[0] - 1, 1), 0.0)


def spectrum_report(X, var_frac=0.95, cond_tol=1e-10):
    """Computes all spectral metrics from a single eigendecomposition of X.
    Valid only on hidden layers, since SIGReg forces the output embedding toward N(0,I)."""
    e = eig_spectrum(X)
    tot = e.sum()
    if tot <= 0:
        return {k: 0.0 for k in ("effective_rank", "participation_ratio",
                                 "stable_rank", f"dims_to_{int(var_frac*100)}",
                                 "log_cond", "inv_eig_sum", "trace")}
    p = e / tot
    p_nz = p[p > 0]
    keep = e > cond_tol * e[0]
    return {
        "effective_rank":  float(np.exp(-(p_nz * np.log(p_nz)).sum())),
        "participation_ratio": float(tot ** 2 / (e ** 2).sum()),
        "stable_rank":     float(tot / e[0]),
        f"dims_to_{int(var_frac*100)}": int(np.searchsorted(np.cumsum(p), var_frac) + 1),
        "log_cond":        float(np.log10(e[0] / e[keep][-1])) if keep.sum() > 1 else 0.0,
        "inv_eig_sum":     float((1.0 / e[keep]).sum()),
        "trace":           float(tot),
        "n_kept":          int(keep.sum()),
    }


def knn_accuracy(Z_tr, y_tr, Z_te, y_te, k=20, metric="cosine", chunk=512):
    """Computes k-nearest-neighbor classification accuracy between train and test embeddings.
    Processes test points in chunks and supports cosine or euclidean distance metrics."""
    Z_tr = np.asarray(Z_tr, np.float32); Z_te = np.asarray(Z_te, np.float32)
    y_tr = np.asarray(y_tr).ravel();     y_te = np.asarray(y_te).ravel()
    if metric == "cosine":
        Z_tr = Z_tr / (np.linalg.norm(Z_tr, axis=1, keepdims=True) + 1e-12)
        Z_te = Z_te / (np.linalg.norm(Z_te, axis=1, keepdims=True) + 1e-12)
    n_cls = int(max(y_tr.max(), y_te.max())) + 1
    correct = 0
    for i in range(0, len(Z_te), chunk):
        B = Z_te[i:i + chunk]
        if metric == "cosine":
            d = -(B @ Z_tr.T)
        else:
            d = ((B ** 2).sum(1)[:, None] - 2 * B @ Z_tr.T + (Z_tr ** 2).sum(1)[None])
        idx = np.argpartition(d, k, axis=1)[:, :k]
        votes = np.zeros((len(B), n_cls), np.int32)
        lab = y_tr[idx]
        for c in range(n_cls):
            votes[:, c] = (lab == c).sum(1)
        correct += (votes.argmax(1) == y_te[i:i + chunk]).sum()
    return float(correct / len(Z_te))


def twonn_dimension(X, discard_frac=0.1, max_n=4000, seed=0):
    """Estimates intrinsic dimension via the TwoNN method (Facco et al. 2017), catching curved low-dim manifolds.
    Subsamples to max_n points since it needs the full pairwise distance matrix."""
    X = np.asarray(X, np.float64)
    if len(X) > max_n:
        X = X[np.random.default_rng(seed).choice(len(X), max_n, replace=False)]
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] - 2 * X @ X.T + sq[None], 0.0))
    np.fill_diagonal(D, np.inf)
    D.sort(axis=1)
    r1, r2 = D[:, 0], D[:, 1]
    ok = (r1 > 0) & np.isfinite(r2)
    mu = r2[ok] / r1[ok]
    mu = np.sort(mu[mu > 1.0])
    n = len(mu)
    if n < 10:
        return float("nan")
    keep = int(n * (1 - discard_frac))
    mu = mu[:keep]
    F = np.arange(1, keep + 1) / n          # empirical CDF
    x = np.log(mu); y = -np.log(np.maximum(1 - F, 1e-12))
    return float((x @ y) / (x @ x))         # regression through the origin


def ks_two_sample(a, b):
    """Computes the two-sample Kolmogorov-Smirnov statistic between arrays a and b.
    Returns the maximum absolute difference between their empirical CDFs."""
    a = np.sort(np.asarray(a, np.float64)); b = np.sort(np.asarray(b, np.float64))
    v = np.concatenate([a, b])
    ca = np.searchsorted(a, v, side="right") / len(a)
    cb = np.searchsorted(b, v, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def projected_two_sample(Z1, Z2, A):
    """Projects Z1 and Z2 onto shared directions A and runs a per-direction KS two-sample test.
    Returns the max, mean, and 90th percentile of the KS statistics across directions."""
    P1 = np.asarray(Z1) @ np.asarray(A)
    P2 = np.asarray(Z2) @ np.asarray(A)
    d = np.array([ks_two_sample(P1[:, m], P2[:, m]) for m in range(A.shape[1])])
    return {"ks_max": float(d.max()), "ks_mean": float(d.mean()),
            "ks_q90": float(np.quantile(d, 0.9))}


def collapse_report(Z):
    """Computes cheap collapse diagnostics (mean, std, effective rank) to flag a collapsed encoder before costlier metrics.
    A collapsed arm shows near-zero std and effective rank near 1."""
    Z = np.asarray(Z, np.float64)
    e = eig_spectrum(Z)
    p = e / e.sum() if e.sum() > 0 else e
    p_nz = p[p > 0]
    return {
        "mean_abs_mean": float(np.abs(Z.mean(0)).mean()),
        "mean_std":      float(Z.std(0).mean()),
        "min_std":       float(Z.std(0).min()),
        "effective_rank": float(np.exp(-(p_nz * np.log(p_nz)).sum())) if len(p_nz) else 0.0,
        "dim":           int(Z.shape[1]),
    }
