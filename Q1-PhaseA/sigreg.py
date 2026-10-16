from functools import partial
import jax
import jax.numpy as jnp

_trapz = getattr(jnp, "trapezoid", None) or jnp.trapz


# ---------------------------------------------------------------- directions
# Algorithm 1 slice sampling. Returns (K, M) with unit-norm COLUMNS.
# Direction set must stay FIXED for whole PC relaxation: resampling mid-inference
# changes the energy function being descended, so fixed point wouldn't exist.
# Resample once per weight update (paper: resampling per step is a free coverage gain).
def sample_directions(key, K, M):
    A = jax.random.normal(key, (K, M))
    return A / jnp.linalg.norm(A, axis=0, keepdims=True)


# ------------------------------------------------------------- Epps-Pulley
# Directional Epps-Pulley statistic.
# P: (N,M) projections a_m^T z_n, NOT standardised (tests against N(0,1) incl. mean/var).
# Standardising here would defeat the regulariser since it constrains scale too.
# Returns (M,) statistic per direction.
# Uses real identity |ecf-w|^2 = (mean cos(tP)-w)^2 + (mean sin(tP))^2, avoids complex dtype under jax.grad.
@partial(jax.jit, static_argnames=("n_points", "scale_by_n"))
def epps_pulley(P, t_max=5.0, n_points=17, scale_by_n=True):
    N = P.shape[0]
    t = jnp.linspace(-t_max, t_max, n_points)          # (T,)
    w = jnp.exp(-0.5 * t ** 2)                         # (T,) CF of N(0,1) == window
    xt = P[:, :, None] * t                             # (N, M, T)
    re = jnp.cos(xt).mean(axis=0)                      # (M, T)
    im = jnp.sin(xt).mean(axis=0)                      # (M, T)
    err = ((re - w) ** 2 + im ** 2) * w                # (M, T) weighted L2
    stat = _trapz(err, t, axis=1)                      # (M,)
    return stat * N if scale_by_n else stat


# Definition 2. Z: (N, K) embeddings, A: (K, M) directions -> scalar.
@partial(jax.jit, static_argnames=("n_points", "scale_by_n"))
def sigreg(Z, A, t_max=5.0, n_points=17, scale_by_n=True):
    return epps_pulley(Z @ A, t_max, n_points, scale_by_n).mean()


# dSIGReg/dZ, shape (N, K).
# Error term for PC output node when loss is not per-sample.
# Depends on WHOLE batch: row n depends on every other row.
# This batch-coupling is what the two output_drive variants differ on.
def sigreg_grad(Z, A, **kw):
    return jax.grad(lambda z: sigreg(z, A, **kw))(Z)


# ------------------------------------------------------- invariance loss
# LeJEPA prediction loss, Eq (7) form. Zv: (V,B,K); returns scalar mean over batch.
# mu_n = (1/V_g) sum_{v<V_g} z_{n,v}; L_n = (1/V) sum_v' ||mu_n - z_{n,v'}||^2
# Eq (7) (distance-from-centre) != Eq (5) (all-pairs) despite paper App. B.6 claim;
# verified numerically they differ by within-global-view variance (dropped cross terms).
# We define loss as Eq (7), matching reference code; do not re-derive from B.6.
@partial(jax.jit, static_argnames=("n_global",))
def prediction_loss(Zv, n_global):
    mu = Zv[:n_global].mean(axis=0)                    # (B, K)
    # FIX (2026-09): reduce with mean over the FEATURE dim, not sum.
    # Reference impl (LeJEPA MINIMAL.md L172) is
    #     inv_loss = (proj.mean(0) - proj).square().mean()
    # i.e. mean over views, batch AND features.  Summing over K made this
    # term embed_dim (=128) times larger than SIGReg, which drove the
    # backprop arm into total collapse.
    return ((mu[None] - Zv) ** 2).mean()


# Parameterless BatchNorm over pooled (V*B) axis, per feature.
# Scale must be quotiented out: without it, Z=0 is exact fixed pt of
# Epps-Pulley stat (grad=0 at Z=0); verified: stat=104.6837, grad=0.0,
# LR sweep froze there. Pooled over views (not per-view) so cross-view
# diffs invariance term targets aren't removed; no learned params to keep arms symmetric.
def batchnorm_views(Zv, eps=1e-5):
    flat = Zv.reshape(-1, Zv.shape[-1])
    mu = flat.mean(axis=0)
    var = flat.var(axis=0)
    return (Zv - mu) / jnp.sqrt(var + eps)


# Total Phase A objective.
# L = lambd * (1/V) sum_v SIGReg(Z_v, A) + (1 - lambd) * L_pred
# use_sigreg=False gives the invariance-only collapse control.
# Returns (total, aux_dict).
def lejepa_loss(Zv, A, n_global, lambd, use_sigreg=True, projector="bn", **sig_kw):
    if projector == "bn":
        Zv = batchnorm_views(Zv)
    pred = prediction_loss(Zv, n_global)
    if use_sigreg:
        sig = jnp.mean(jnp.stack([sigreg(Z, A, **sig_kw) for Z in Zv]))
        total = lambd * sig + (1.0 - lambd) * pred
    else:
        sig = jnp.array(0.0)
        total = pred
    return total, {"pred": pred, "sigreg": sig, "total": total}
