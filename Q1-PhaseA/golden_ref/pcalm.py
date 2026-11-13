# PC-ALM -- augmented-Lagrangian predictive coding for the Phase A encoder. Adds one Lagrange
# multiplier ("dual") per layer/sample/view to the PC relaxation; alpha=0 recovers plain PC
# exactly. Ref: Seely & Gould, "Augmented Lagrangian Predictive Coding" (arXiv:2605.31022v1).
# cfg.pc.eta_h is the PER-SAMPLE step, multiplied by B here (the old activity_lr wasn't,
# which stalled relaxation at batch_size=256 -- see diag_inference.py).
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
    """Returns each layer's prediction f_l(z_{l-1}), mirroring jpc.pc_energy_fn's terms exactly.
    Keep in sync with jpc; re-run test_al_energy_matches_jpc_at_zero_duals after any edit."""
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
    """Returns the augmented Lagrangian energy (paper Eq. 9, completing-the-square form).
    duals=None gives the plain PC energy, matching jpc.pc_energy_fn exactly at rho=1."""
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
        """Returns the LeJEPA loss for a stack of per-view embeddings Zv."""
        return S.lejepa_loss(
            Zv, A, cfg.data.n_global_views, cfg.sig.lambd, cfg.use_sigreg,
            projector=cfg.projector, t_max=cfg.sig.t_max,
            n_points=cfg.sig.n_points, scale_by_n=cfg.sig.scale_by_n)[0]
    return loss_z


def relax(params, Zin, A, cfg: RunCfg, key):
    """Runs T primal steps with T-1 interleaved dual updates (Algorithm 1) and returns (acts, duals, history).
    At cfg.pc.alpha == 0 the duals stay zero and this reduces to the plain PC relaxation."""
    V, B = Zin.shape[0], Zin.shape[1]
    alpha, rho, T = cfg.pc.alpha, cfg.pc.rho, cfg.pc.T
    # the *B that turns cfg.pc.eta_h into a per-sample step -- see header comment above
    eff_lr = cfg.pc.eta_h * B

    acts0 = []
    for v in range(V):
        z, hs = E.forward(params, Zin[v], cfg.enc)
        acts0.append([*hs, z])
    duals0 = zero_duals(acts0)

    loss_z = _objective(cfg, A)

    def out_stack(acts):
        """Stacks the final-layer activity across views."""
        return jnp.stack([acts[v][-1] for v in range(V)])

    def F_total(acts, duals):
        """Returns the augmented Lagrangian energy summed over all views."""
        return sum(al_energy(params, acts[v], Zin[v], duals[v], rho, cfg.enc,
                             cfg.pc.out_scale) for v in range(V))

    # "frozen" caches the output drive from the feedforward embeddings; "coupled"
    # recomputes it every step.  Unchanged from the previous implementation.
    g0 = jax.grad(loss_z)(out_stack(acts0)) if cfg.pc.output_drive == "frozen" else None
    dF = jax.grad(F_total, argnums=0)
    dL = jax.grad(lambda a: loss_z(out_stack(a)))

    def primal(acts, duals):
        """Returns acts after one gradient-descent step on the (possibly output-coupled) energy."""
        g = dF(acts, duals)
        if cfg.pc.output_drive == "coupled":
            gL = dL(acts)
            g = [[gi + gj for gi, gj in zip(g[v], gL[v])] for v in range(V)]
        else:
            g = [[*g[v][:-1], g[v][-1] + g0[v]] for v in range(V)]
        return [[z - eff_lr * gz for z, gz in zip(acts[v], g[v])]
                for v in range(V)]

    def dual(acts, duals):
        """Returns duals after one dual-ascent step on the residuals (no-op when alpha == 0)."""
        if alpha == 0.0:
            return duals
        r = [residuals(params, acts[v], Zin[v], cfg.enc) for v in range(V)]
        return [[lam + alpha * ri for lam, ri in zip(duals[v], r[v])]
                for v in range(V)]

    def body(carry, _):
        """Runs one primal+dual cycle for jax.lax.scan, returning the new carry and (F, L) history."""
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
    """Returns the weight gradient of the augmented Lagrangian at the relaxed (stop-gradient) primal-dual state.
    Each weight sees only its own layer's composite credit rho * r_l + lam_l, same as pc-alm's method_grad."""
    import equinox as eqx

    acts, duals, hist = relax(params, Zin, A, cfg, key)
    acts = jax.lax.stop_gradient(acts)
    duals = jax.lax.stop_gradient(duals)
    V = Zin.shape[0]

    def F_of_params(p):
        """Returns the augmented Lagrangian energy at weights p, with acts/duals fixed."""
        return sum(al_energy(p, acts[v], Zin[v], duals[v], cfg.pc.rho, cfg.enc,
                             cfg.pc.out_scale) for v in range(V))

    g = eqx.filter_grad(F_of_params)(params)
    return g, {"F_final": hist["F"][-1], "L_final": hist["L"][-1],
               "F_hist": hist["F"], "L_hist": hist["L"],
               "dual_norm": jnp.sqrt(sum(jnp.sum(l ** 2)
                                         for dv in duals for l in dv))}
