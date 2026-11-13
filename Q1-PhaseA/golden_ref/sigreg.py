# SIGReg -- Sketched Isotropic Gaussian Regularization: a JAX port of LeJEPA Algorithm 1 (Epps-Pulley
# variant) plus the prediction/invariance loss. Definition 2 averages the statistic over directions (not
# the max); Algorithm 1 uses O(N) numerical quadrature of the empirical vs. N(0,1) characteristic function,
# not the classical O(N^2) closed form. Known gap: the paper listing truncates Algorithm 1's final scale
# factor; scale_by_n assumes it is N, which matters only for an absolute statistic (it folds into lambd otherwise).
from functools import partial
import jax
import jax.numpy as jnp

_trapz = getattr(jnp, "trapezoid", None) or jnp.trapz


# ---------------------------------------------------------------- directions
def sample_directions(key, K, M):
    """Samples Algorithm 1's K unit-norm direction columns (K, M); must stay fixed for one PC relaxation
    (resampling mid-inference would change the energy the activities are descending)."""
    A = jax.random.normal(key, (K, M))
    return A / jnp.linalg.norm(A, axis=0, keepdims=True)


# ------------------------------------------------------------- Epps-Pulley
@partial(jax.jit, static_argnames=("n_points", "scale_by_n"))
def epps_pulley(P, t_max=5.0, n_points=17, scale_by_n=True):
    """Computes the directional Epps-Pulley statistic on unstandardised projections P (N, M) -> (M,),
    via the real-valued cos/sin identity (avoids complex dtypes under jax.grad)."""
    N = P.shape[0]
    t = jnp.linspace(-t_max, t_max, n_points)          # (T,)
    w = jnp.exp(-0.5 * t ** 2)                         # (T,) CF of N(0,1) == window
    xt = P[:, :, None] * t                             # (N, M, T)
    re = jnp.cos(xt).mean(axis=0)                      # (M, T)
    im = jnp.sin(xt).mean(axis=0)                      # (M, T)
    err = ((re - w) ** 2 + im ** 2) * w                # (M, T) weighted L2
    stat = _trapz(err, t, axis=1)                      # (M,)
    return stat * N if scale_by_n else stat


@partial(jax.jit, static_argnames=("n_points", "scale_by_n"))
def sigreg(Z, A, t_max=5.0, n_points=17, scale_by_n=True):
    """Computes SIGReg (Definition 2): mean Epps-Pulley statistic of Z's projections onto directions A."""
    return epps_pulley(Z @ A, t_max, n_points, scale_by_n).mean()


def sigreg_grad(Z, A, **kw):
    """Returns dSIGReg/dZ (N, K), the batch-coupled error term used as the PC output node's drive."""
    return jax.grad(lambda z: sigreg(z, A, **kw))(Z)


# ------------------------------------------------------- invariance loss
@partial(jax.jit, static_argnames=("n_global",))
def prediction_loss(Zv, n_global):
    """Computes the LeJEPA prediction loss (Eq. 7): mean squared distance of each view's embedding from
    the global-view centroid. Uses Eq (7) as defined; the paper's claimed Eq (5) equivalence does not hold here."""
    mu = Zv[:n_global].mean(axis=0)                    # (B, K)
    # FIX (2026-09): mean over the FEATURE dim too, matching the reference (LeJEPA MINIMAL.md L172):
    # inv_loss = (proj.mean(0) - proj).square().mean() -- mean over views, batch AND features.
    # Summing over K made this term 128x larger than SIGReg, driving the backprop arm into collapse.
    return ((mu[None] - Zv) ** 2).mean()


def batchnorm_views(Zv, eps=1e-5):
    """Applies parameterless BatchNorm over the pooled (V*B) axis per feature, matching LeJEPA's norm_layer.
    Without it, total collapse is an exact fixed point of SIGReg (zero gradient), unescapable by any learning rate."""
    flat = Zv.reshape(-1, Zv.shape[-1])
    mu = flat.mean(axis=0)
    var = flat.var(axis=0)
    return (Zv - mu) / jnp.sqrt(var + eps)


def lejepa_loss(Zv, A, n_global, lambd, use_sigreg=True, projector="bn", **sig_kw):
    """Computes the total Phase A objective lambd*SIGReg + (1-lambd)*prediction_loss (or prediction-only
    if use_sigreg=False); returns (total, aux_dict)."""
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
