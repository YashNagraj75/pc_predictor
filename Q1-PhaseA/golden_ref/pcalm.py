"""PC-ALM -- augmented-Lagrangian predictive coding for the Phase A encoder.

Adds one Lagrange multiplier ("dual") per layer, per sample, per view to the PC
relaxation.  `alpha = 0` recovers plain PC exactly, so this module is a strict
generalisation of the old `train.pc_relax` and the alpha=0 path is a built-in
control rather than a separate code path to keep in sync.

Reference: Seely & Gould, "Augmented Lagrangian Predictive Coding"
(arXiv:2605.31022v1); reference implementation github.com/SakanaAI/pc-alm.

    PC        :  z <- z - eta * dF/dz                     (F = PC energy)
    PC-ALM    :  z <- z - eta * dL/dz,  then  lam += alpha * r
                 where L is the augmented Lagrangian and r the residual.

At the fixed point the activities return to the feedforward pass (r -> 0) and
lam has integrated to the backprop adjoint at that layer, so the weight update
computed from L becomes the backprop weight gradient.


Why the residuals are rebuilt here instead of taken from jpc
------------------------------------------------------------
PC-ALM needs the per-layer residual r_l = z_l - f_l(z_{l-1}) EXPLICITLY.
`jpc.pc_energy_fn` only returns the energy -- the squared norm -- from which
r_l cannot be recovered even with `record_layers=True`.  So we have to rebuild
the prediction f_l.

To stay consistent with jpc rather than re-deriving its skip/scaling
convention by hand (an earlier hand-derivation of exactly this got the
convention wrong and produced nonsense residuals), `layer_predictions` calls
jpc's own `model`, `skip_model` and `_get_param_scalings`, and mirrors the term
structure of `pc_energy_fn` exactly:

    layer 0       : pred = scalings[0]  * model[0](x)            -- no skip
    layers 1..L-2 : pred = scalings[l]  * model[l](z_{l-1})
                           + skip_l(z_{l-1})                     -- skip applied
    layer L-1     : pred = scalings[-1] * model[-1](z_{L-2})     -- no skip

`test_pcalm.py::test_al_energy_matches_jpc_at_zero_duals` pins this: with
duals=0 and rho=1, `al_energy` must equal `E.energy` (i.e. `jpc.pc_energy_fn`)
to floating-point tolerance.  Do NOT edit `layer_predictions` without
re-running that test.


The step size
-------------
`jpc.pc_energy_fn` returns the energy divided by the batch size (see its
docstring: "normalised by the batch size").  One sample's activities appear in
only that sample's energy term, so dF/dz for sample j is 1/B of sample j's own
gradient, and a step size applied directly to it realises eta/B per sample.

`cfg.pc.eta_h` is therefore defined as the PER-SAMPLE step and multiplied by B
here.  This is what the old `cfg.pc.activity_lr` failed to do: at
activity_lr=0.5, batch_size=256 the realised step was 0.002 against a stable
optimum near 0.25, so the relaxation never left its initialisation and credit
never reached the deep layers.  See `diag_inference.py`.
"""
from typing import List, Optional

import jax
import jax.numpy as jnp
import jpc
from jax import vmap

import encoder as E
import sigreg as S
from config import EncoderCfg, RunCfg


# --------------------------------------------------------------- residuals


def layer_predictions(params, acts: List[jnp.ndarray], x, enc: EncoderCfg):
    """Per-layer prediction f_l(z_{l-1}), in the same order as `acts`.

    Mirrors the term structure of `jpc.pc_energy_fn` term for term -- see the
    module docstring.  `acts` is [z_1, ..., z_{L-1}, z_out] with z_0 clamped to
    `x` and not included, which is the order `encoder.forward` returns and the
    order the relaxation carries.
    """
    model, skip = params
    scalings = jpc._get_param_scalings(
        model=model,
        input=x,
        skip_model=skip,
        param_type=enc.parameterisation,
        gamma=E.gamma_for(enc),
    )
    L = len(model)
    preds = []
    for l in range(L):
        prev = x if l == 0 else acts[l - 1]
        pred = scalings[l] * vmap(model[l])(prev)
        # jpc applies skips only to the interior terms: its output term never
        # subtracts a skip, and neither does its first-layer term.
        if 0 < l < L - 1 and skip is not None and skip[l] is not None:
            pred = pred + vmap(skip[l])(prev)
        preds.append(pred)
    return preds


def residuals(params, acts, x, enc: EncoderCfg):
    """r_l = z_l - f_l(z_{l-1}) for every layer, same shapes as `acts`."""
    return [a - p for a, p in zip(acts, layer_predictions(params, acts, x, enc))]


def zero_duals(acts):
    """Duals start at zero (Algorithm 1 line 5), one per activity."""
    return [[jnp.zeros_like(z) for z in acts_v] for acts_v in acts]


# ------------------------------------------------------------------ energy


def al_energy(params, acts, x, duals, rho: float, enc: EncoderCfg,
              out_scale: Optional[float] = None):
    """Augmented Lagrangian in the completing-the-square form (paper Eq. 9).

        L_rho = (1 / 2B) * rho * sum_l || r_l + lam_l / rho ||^2

    `duals=None` gives the plain PC energy, so at rho=1 this reproduces
    `jpc.pc_energy_fn` exactly (pinned by test_pcalm.py).

    The paper's Eq. 9 also carries a -1/(2 rho) sum_l ||lam_l||^2 term, dropped
    here: it depends on neither the activities nor the weights, so it changes
    neither the inference gradient nor the weight gradient.

    NOTE there is no supervised-loss term.  Our objective is LeJEPA, which is
    not a per-sample target -- it enters as a gradient injected at the output
    node during the relaxation (`output_drive`), exactly as in the previous PC
    implementation.  The Lagrangian's boundary condition is therefore supplied
    by that injected gradient rather than by a term in this function.
    """
    r = residuals(params, acts, x, enc)
    if duals is not None:
        r = [ri + li / rho for ri, li in zip(r, duals)]
    scale = 1.0 if out_scale is None else out_scale
    # the final term carries the output-layer precision, matching jpc's
    # `output_energy_scaling`; all others are unweighted.
    terms = [jnp.sum(ri ** 2) for ri in r]
    total = sum(terms[:-1]) + scale * terms[-1]
    return 0.5 * rho * total / x.shape[0]


# -------------------------------------------------------------- relaxation


def _objective(cfg: RunCfg, A):
    """The LeJEPA objective on a stack of per-view embeddings."""
    def loss_z(Zv):
        return S.lejepa_loss(
            Zv, A, cfg.data.n_global_views, cfg.sig.lambd, cfg.use_sigreg,
            projector=cfg.projector, t_max=cfg.sig.t_max,
            n_points=cfg.sig.n_points, scale_by_n=cfg.sig.scale_by_n)[0]
    return loss_z


def relax(params, Zin, A, cfg: RunCfg, key):
    """Primal-dual relaxation.  Returns (acts, duals, history).

    Algorithm 1 of the paper: `T-1` cycles of {one primal step, one dual step},
    then a final primal step.  That is T primal steps in total -- the same
    activity budget as the previous PC implementation -- plus T-1 dual updates.

    At `cfg.pc.alpha == 0` the duals stay zero, `al_energy` collapses to the PC
    energy, and this reduces to exactly the old relaxation.
    """
    V, B = Zin.shape[0], Zin.shape[1]
    alpha, rho, T = cfg.pc.alpha, cfg.pc.rho, cfg.pc.T
    # the *B that turns cfg.pc.eta_h into a per-sample step -- see module docstring
    eff_lr = cfg.pc.eta_h * B

    acts0 = []
    for v in range(V):
        z, hs = E.forward(params, Zin[v], cfg.enc)
        acts0.append([*hs, z])
    duals0 = zero_duals(acts0)

    loss_z = _objective(cfg, A)

    def out_stack(acts):
        return jnp.stack([acts[v][-1] for v in range(V)])

    def F_total(acts, duals):
        return sum(al_energy(params, acts[v], Zin[v], duals[v], rho, cfg.enc,
                             cfg.pc.out_scale) for v in range(V))

    # "frozen" caches the output drive from the feedforward embeddings; "coupled"
    # recomputes it every step.  Unchanged from the previous implementation.
    g0 = jax.grad(loss_z)(out_stack(acts0)) if cfg.pc.output_drive == "frozen" else None
    dF = jax.grad(F_total, argnums=0)
    dL = jax.grad(lambda a: loss_z(out_stack(a)))

    def primal(acts, duals):
        g = dF(acts, duals)
        if cfg.pc.output_drive == "coupled":
            gL = dL(acts)
            g = [[gi + gj for gi, gj in zip(g[v], gL[v])] for v in range(V)]
        else:
            g = [[*g[v][:-1], g[v][-1] + g0[v]] for v in range(V)]
        return [[z - eff_lr * gz for z, gz in zip(acts[v], g[v])]
                for v in range(V)]

    def dual(acts, duals):
        if alpha == 0.0:
            return duals
        r = [residuals(params, acts[v], Zin[v], cfg.enc) for v in range(V)]
        return [[lam + alpha * ri for lam, ri in zip(duals[v], r[v])]
                for v in range(V)]

    def body(carry, _):
        acts, duals = carry
        acts = primal(acts, duals)
        duals = dual(acts, duals)
        return (acts, duals), (F_total(acts, duals), loss_z(out_stack(acts)))

    if T > 1:
        (acts, duals), (F_h, L_h) = jax.lax.scan(
            body, (acts0, duals0), xs=None, length=T - 1)
    else:
        acts, duals, F_h, L_h = acts0, duals0, None, None

    acts = primal(acts, duals)                       # Algorithm 1 line 10
    F_last = F_total(acts, duals)
    L_last = loss_z(out_stack(acts))
    duals_final = dual(acts, duals)

    if F_h is None:
        F_hist, L_hist = jnp.array([F_last]), jnp.array([L_last])
    else:
        F_hist = jnp.concatenate([F_h, jnp.array([F_last])])
        L_hist = jnp.concatenate([L_h, jnp.array([L_last])])

    # `weight_credit_timing`: "pre_dual" uses the duals the final primal step
    # actually saw (paper Algorithm 1); "post_dual" folds in one more dual
    # update.  pc-alm defaults to pre_dual.
    duals_w = duals if cfg.pc.weight_credit_timing == "pre_dual" else duals_final
    return acts, duals_w, {"F": F_hist, "L": L_hist}


def grads(params, Zin, A, cfg: RunCfg, key):
    """Weight gradient from the relaxed primal-dual state.

    Both the activities and the duals are held fixed (stop_gradient) and the
    augmented Lagrangian is differentiated with respect to the weights only --
    the same pattern the previous `pc_grads` used, and the same as pc-alm's
    `method_grad`.  The resulting update is local: each weight sees only its
    own layer's composite credit `rho * r_l + lam_l` and the activity below it.
    """
    import equinox as eqx

    acts, duals, hist = relax(params, Zin, A, cfg, key)
    acts = jax.lax.stop_gradient(acts)
    duals = jax.lax.stop_gradient(duals)
    V = Zin.shape[0]

    def F_of_params(p):
        return sum(al_energy(p, acts[v], Zin[v], duals[v], cfg.pc.rho, cfg.enc,
                             cfg.pc.out_scale) for v in range(V))

    g = eqx.filter_grad(F_of_params)(params)
    return g, {"F_final": hist["F"][-1], "L_final": hist["L"][-1],
               "F_hist": hist["F"], "L_hist": hist["L"],
               "dual_norm": jnp.sqrt(sum(jnp.sum(l ** 2)
                                         for dv in duals for l in dv))}
