"""SIGReg -- Sketched Isotropic Gaussian Regularization.

Faithful JAX port of LeJEPA Algorithm 1 (Epps-Pulley variant) plus the
LeJEPA prediction/invariance loss.

Provenance of every design choice below is the paper, not invention:
  * Definition 2 aggregates the directional statistic by an AVERAGE over
    |A| directions, not the maximum of Theorem 2 -- the paper states this is
    to avoid sparse gradients over the directions.
  * Algorithm 1 computes the Epps-Pulley statistic by NUMERICAL QUADRATURE of
    the weighted L2 distance between the empirical characteristic function and
    the N(0,1) characteristic function, over `n_points` points on
    [-t_max, t_max].  It is NOT the closed-form double sum found in the
    classical Epps-Pulley literature.  Cost is O(N), not O(N^2).
  * exp(-t^2/2) plays two roles at once: it is the CF of N(0,1) AND the
    weighting window w(t).
  * Directions are unit-norm COLUMNS of A, shape (K, M).

Known gap, stated honestly: the PDF's rendering of Algorithm 1 truncates the
final line (`T = torch.trapz(err, t, dim=1) * ...`).  The trailing factor is
almost certainly N -- classical EP is  n * integral |phi_n - phi|^2 w  -- and
`scale_by_n` implements that.  For a REGULARISER the factor is a constant that
folds into `lambd`, so nothing downstream depends on resolving it.  Check
against the official codebase before publishing an absolute statistic value.
"""
from functools import partial
import jax
import jax.numpy as jnp

_trapz = getattr(jnp, "trapezoid", None) or jnp.trapz


# ---------------------------------------------------------------- directions
def sample_directions(key, K, M):
    """Algorithm 1 slice sampling.  Returns (K, M) with unit-norm COLUMNS.

    A direction set must be held FIXED for the whole of a PC relaxation:
    resampling mid-inference changes the energy function the activities are
    descending, so the fixed point would not exist.  Resample once per weight
    update (the paper reports resampling per step is a free coverage gain).
    """
    A = jax.random.normal(key, (K, M))
    return A / jnp.linalg.norm(A, axis=0, keepdims=True)


# ------------------------------------------------------------- Epps-Pulley
@partial(jax.jit, static_argnames=("n_points", "scale_by_n"))
def epps_pulley(P, t_max=5.0, n_points=17, scale_by_n=True):
    """Directional Epps-Pulley statistic.

    P : (N, M) projections  a_m^T z_n   -- NOT standardised.  SIGReg tests
        against N(0,1) including mean and variance, so it constrains scale
        too; standardising here would defeat the regulariser.
    returns : (M,) statistic per direction.

    Implemented with the real identity
        |ecf - w|^2 = (mean cos(tP) - w)^2 + (mean sin(tP))^2
    which avoids complex dtypes under jax.grad and is exactly equivalent.
    """
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
    """Definition 2.  Z: (N, K) embeddings, A: (K, M) directions -> scalar."""
    return epps_pulley(Z @ A, t_max, n_points, scale_by_n).mean()


def sigreg_grad(Z, A, **kw):
    """dSIGReg/dZ, shape (N, K).

    This is the object the PC output node needs as its error term when the
    loss is not a per-sample target.  It depends on the WHOLE batch: row n of
    the result is a function of every other row.  That is the batch-coupling
    the two `output_drive` variants differ on.
    """
    return jax.grad(lambda z: sigreg(z, A, **kw))(Z)


# ------------------------------------------------------- invariance loss
@partial(jax.jit, static_argnames=("n_global",))
def prediction_loss(Zv, n_global):
    """LeJEPA prediction loss, Equation (7) form.

    Zv : (V, B, K) embeddings, views 0..n_global-1 are the global views.
    returns : scalar, mean over the batch.

        mu_n  = (1/V_g) sum_{v<V_g} z_{n,v}
        L_n   = (1/V) sum_{v'} || mu_n - z_{n,v'} ||^2

    IMPORTANT -- this is Eq (7), the distance-from-centre form, which is what
    the reference implementation computes.  The paper's Appendix B.6 asserts
    Eq (5) (the all-pairs form) equals Eq (7); a numerical check earlier in
    this project showed it does NOT -- the two differ by exactly the
    within-global-view variance, because an intermediate step drops
    off-diagonal cross terms.  Nothing here depends on that identity; we take
    Eq (7) as the definition, matching the code.  Do not re-derive from B.6.
    """
    mu = Zv[:n_global].mean(axis=0)                    # (B, K)
    # FIX (2026-09): reduce with mean over the FEATURE dim, not sum.
    # Reference impl (LeJEPA MINIMAL.md L172) is
    #     inv_loss = (proj.mean(0) - proj).square().mean()
    # i.e. mean over views, batch AND features.  Summing over K made this
    # term embed_dim (=128) times larger than SIGReg, which drove the
    # backprop arm into total collapse.
    return ((mu[None] - Zv) ** 2).mean()


def batchnorm_views(Zv, eps=1e-5):
    """Parameterless BatchNorm over the POOLED (V*B) axis, per feature.

    LeJEPA never lets SIGReg see the raw backbone output: Algorithm 1 acts
    on the output of MLP(512,[2048,2048,proj_dim], norm_layer=BatchNorm1d).
    The load-bearing property is that scale is quotiented out, so the
    "shrink everything to zero" solution of the invariance term buys
    nothing.  Without it, total collapse is an EXACT fixed point of the
    Epps-Pulley statistic (gradient identically zero at Z=0), which no
    learning rate escapes -- verified numerically: statistic 104.6837,
    gradient 0.0, and every backprop run in the LR sweep froze there.

    Pooled over views, not per-view: per-view standardisation would remove
    exactly the cross-view differences the invariance term exists to
    reduce.  Parameterless (no learned MLP) so both arms stay symmetric
    and the PC computation graph is unchanged -- this is a differentiable
    function of the output activities, nothing more.
    """
    flat = Zv.reshape(-1, Zv.shape[-1])
    mu = flat.mean(axis=0)
    var = flat.var(axis=0)
    return (Zv - mu) / jnp.sqrt(var + eps)


def lejepa_loss(Zv, A, n_global, lambd, use_sigreg=True, projector="bn", **sig_kw):
    """Total Phase A objective.

    L = lambd * (1/V) sum_v SIGReg(Z_v, A)  +  (1 - lambd) * L_pred

    use_sigreg=False gives the invariance-only collapse control.
    Returns (total, aux_dict).
    """
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
