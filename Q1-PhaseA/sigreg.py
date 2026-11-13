# SIGReg (sliced Epps-Pulley normality test) and the LeJEPA invariance loss:
# the Phase-A objective used to regularise/decorrelate embeddings and keep
# multi-view predictions close without an explicit negative-pairs term.
from functools import partial
import jax
import jax.numpy as jnp

_trapz = getattr(jnp, "trapezoid", None) or jnp.trapz


# ---------------------------------------------------------------- directions
# Algorithm 1 slice sampling. Direction set must stay FIXED for whole PC relaxation
# (resampling mid-inference changes the energy being descended); resample once per
# weight update instead (paper: resampling per step is a free coverage gain).
def sample_directions(key, K, M):
    """Returns K x M random projection directions with unit-norm columns."""
    A = jax.random.normal(key, (K, M))
    return A / jnp.linalg.norm(A, axis=0, keepdims=True)


# ------------------------------------------------------------- Epps-Pulley
# Directional Epps-Pulley statistic on NOT-standardised projections P (tests
# against N(0,1) incl. mean/var; standardising would defeat the regulariser).
# Uses real identity |ecf-w|^2=(mean cos(tP)-w)^2+(mean sin(tP))^2, avoiding complex dtype under jax.grad.
@partial(jax.jit, static_argnames=("n_points", "scale_by_n"))
def epps_pulley(P, t_max=5.0, n_points=17, scale_by_n=True):
    """Returns the per-direction Epps-Pulley normality statistic for projections P."""
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
    """Returns the mean Epps-Pulley statistic of Z projected onto directions A."""
    return epps_pulley(Z @ A, t_max, n_points, scale_by_n).mean()


# dSIGReg/dZ, shape (N, K): error term for PC output node when loss is not
# per-sample. Depends on the WHOLE batch (row n depends on every other row);
# this batch-coupling is what the two output_drive variants differ on.
def sigreg_grad(Z, A, **kw):
    """Returns dSIGReg/dZ, the SIGReg gradient with respect to the embeddings."""
    return jax.grad(lambda z: sigreg(z, A, **kw))(Z)


# ------------------------------------------------------- invariance loss
# LeJEPA prediction loss, Eq (7) form (distance-from-centre, not all-pairs Eq 5,
# despite paper App. B.6 claim -- verified they differ by within-global-view
# variance). Zv: (V,B,K); returns scalar mean over batch, matching reference code.
@partial(jax.jit, static_argnames=("n_global",))
def prediction_loss(Zv, n_global):
    """Returns the mean squared distance of each view's embedding from the global-view mean."""
    mu = Zv[:n_global].mean(axis=0)                    # (B, K)
    # FIX (2026-09): reduce with mean over the FEATURE dim, not sum (ref: LeJEPA
    # MINIMAL.md L172). Summing over K made this term 128x larger than SIGReg,
    # driving the backprop arm into total collapse.
    return ((mu[None] - Zv) ** 2).mean()


# Parameterless BatchNorm over pooled (V*B) axis, per feature. Scale must be
# quotiented out: without it Z=0 is an exact fixed point of the Epps-Pulley stat
# (grad=0 there; verified stat=104.6837, froze an LR sweep). Pooled over views
# (not per-view) so cross-view diffs the invariance term targets aren't removed.
def batchnorm_views(Zv, eps=1e-5):
    """Returns Zv batch-normalised per feature over the pooled view*batch axis."""
    flat = Zv.reshape(-1, Zv.shape[-1])
    mu = flat.mean(axis=0)
    var = flat.var(axis=0)
    return (Zv - mu) / jnp.sqrt(var + eps)


# Total Phase A objective.
# L = lambd * (1/V) sum_v SIGReg(Z_v, A) + (1 - lambd) * L_pred
# use_sigreg=False gives the invariance-only collapse control.
# Returns (total, aux_dict).
def lejepa_loss(Zv, A, n_global, lambd, use_sigreg=True, projector="bn", **sig_kw):
    """Returns the combined (SIGReg, prediction) loss and an aux dict with both components."""
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
