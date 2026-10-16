from typing import List, Optional

import jax
import jax.numpy as jnp
import jpc
from jax import vmap

import encoder as E
import sigreg as S
import sparse as SP
from config import EncoderCfg, RunCfg


# --------------------------------------------------------------- residuals


# Per-layer prediction f_l(z_{l-1}), same order as `acts`.
# Mirrors term structure of jpc.pc_energy_fn, term for term (see module docstring).
# acts = [z_1, ..., z_{L-1}, z_out], z_0 clamped to x and not included.
# Order matches encoder.forward return order and relaxation order.
def layer_predictions(params, acts: List[jnp.ndarray], x, enc: EncoderCfg, norm=None):
    model, skip = params[0], params[1]   # case-C params carry a 3rd entry, D
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
        if norm is not None and l == norm[0]:
            # sparse arms: per-sample normalised prediction at layer s
            pred = SP.normalise_rows(pred, norm[1])     # safe norm, finite grad at 0
        preds.append(pred)
    return preds


# r_l = z_l - f_l(z_{l-1}) for every layer, same shapes as `acts`.
# norm = (s, c) normalises prediction at layer s (sparse arms);
# None leaves every layer exactly as jpc defines it.
def residuals(params, acts, x, enc: EncoderCfg, norm=None):
    return [a - p for a, p in zip(acts, layer_predictions(params, acts, x, enc, norm))]


# Duals start at zero (Algorithm 1 line 5), one per activity.
def zero_duals(acts):
    return [[jnp.zeros_like(z) for z in acts_v] for acts_v in acts]


# ------------------------------------------------------------------ energy


# Augmented Lagrangian, completing-the-square form (paper Eq. 9):
# L_rho = (1/2B) * rho * sum_l || r_l + lam_l/rho ||^2
# duals=None -> plain PC energy; at rho=1 matches jpc.pc_energy_fn (test_pcalm.py)
# Dropped -1/(2 rho) sum_l ||lam_l||^2 term (indep. of activities/weights, no grad effect)
# No supervised-loss term; LeJEPA grad injected at output node via output_drive during relaxation
def al_energy(params, acts, x, duals, rho: float, enc: EncoderCfg,
              out_scale: Optional[float] = None, norm=None):
    r = residuals(params, acts, x, enc, norm)
    if duals is not None:
        r = [ri + li / rho for ri, li in zip(r, duals)]
    scale = 1.0 if out_scale is None else out_scale
    # the final term carries the output-layer precision, matching jpc's
    # `output_energy_scaling`; all others are unweighted.
    terms = [jnp.sum(ri ** 2) for ri in r]
    total = sum(terms[:-1]) + scale * terms[-1]
    return 0.5 * rho * total / x.shape[0]


# -------------------------------------------------------------- relaxation


# LeJEPA objective computed on a stack of per-view embeddings.
def _objective(cfg: RunCfg, A):
    def loss_z(Zv):
        return S.lejepa_loss(
            Zv, A, cfg.data.n_global_views, cfg.sig.lambd, cfg.use_sigreg,
            projector=cfg.projector, t_max=cfg.sig.t_max,
            n_points=cfg.sig.n_points, scale_by_n=cfg.sig.scale_by_n)[0]
    return loss_z


# Primal-dual relaxation. Returns (acts, duals, history).
# Algorithm 1: T-1 cycles of {primal step, dual step}, then final primal step.
# T primal steps total (same budget as old PC) plus T-1 dual updates.
# If cfg.pc.alpha == 0, duals stay zero, al_energy collapses to PC energy,
# reduces exactly to old relaxation.
def relax(params, Zin, A, cfg: RunCfg, key):
    V, B = Zin.shape[0], Zin.shape[1]
    alpha, rho, T = cfg.pc.alpha, cfg.pc.rho, cfg.pc.T
    # the *B that turns cfg.pc.eta_h into a per-sample step -- see module docstring
    eff_lr = cfg.pc.eta_h * B

    sparse_on = SP.active(cfg)
    acts0, a0 = [], None
    if not sparse_on:
        for v in range(V):
            z, hs = E.forward(params, Zin[v], cfg.enc)
            acts0.append([*hs, z])
    else:
        # sparse arms start from the DEPLOYED forward pass (case C at t = 0)
        assert cfg.pc.init == "feedforward", "sparse arms need feedforward init"
        a0 = []
        for v in range(V):
            acts_v, a_v = SP.relax_init(params, Zin[v], cfg)
            acts0.append(acts_v)
            a0.append(a_v)
        if cfg.sparse.case not in ("C", "D"):
            a0 = None
    alphas = SP.layer_alphas(cfg, len(acts0[0]))
    norm = SP.norm_spec(cfg)                 # None for case none
    # per-layer EFFECTIVE step (per-sample step x B); case B uses its own step
    # at layer s, s_step / rho (the prox threshold follows in SP.prox_zs)
    lrs = [eff_lr] * len(acts0[0])
    if cfg.sparse.case == "B":
        lrs[cfg.sparse.layer] = B * cfg.sparse.s_step / cfg.pc.rho
    # case C code step, once per relaxation (D is constant within it)
    eta_g = SP.code_step(params, cfg) if cfg.sparse.case in ("C", "D") else None
    duals0 = zero_duals(acts0)

    loss_z = _objective(cfg, A)

    def out_stack(acts):
        return jnp.stack([acts[v][-1] for v in range(V)])

    def F_total(acts, duals):
        return sum(al_energy(params, acts[v], Zin[v], duals[v], rho, cfg.enc,
                             cfg.pc.out_scale, norm) for v in range(V))

    # smooth part (differentiated) and reported energy (+ the L1 terms)
    def F_smooth(acts, duals, a):
        if not sparse_on:
            return F_total(acts, duals)
        return F_total(acts, duals) + sum(
            SP.head_smooth(params, acts[v], None if a is None else a[v], cfg, x=Zin[v])
            for v in range(V))

    def F_report(acts, duals, a):
        if not sparse_on:
            return F_total(acts, duals)
        return F_smooth(acts, duals, a) + sum(
            SP.nonsmooth(acts[v], None if a is None else a[v], cfg) for v in range(V))

    # "frozen" caches the output drive from the feedforward embeddings; "coupled"
    # recomputes it every step.  Unchanged from the previous implementation.
    g0 = jax.grad(loss_z)(out_stack(acts0)) if cfg.pc.output_drive == "frozen" else None
    dF = jax.grad(F_total, argnums=0) if not sparse_on else jax.grad(F_smooth, argnums=0)
    dL = jax.grad(lambda a: loss_z(out_stack(a)))

    def primal(acts, duals, a=None):
        g = dF(acts, duals) if not sparse_on else dF(acts, duals, a)
        if cfg.pc.output_drive == "coupled":
            gL = dL(acts)
            g = [[gi + gj for gi, gj in zip(g[v], gL[v])] for v in range(V)]
        else:
            g = [[*g[v][:-1], g[v][-1] + g0[v]] for v in range(V)]
        if not sparse_on:
            return [[z - eff_lr * gz for z, gz in zip(acts[v], g[v])]
                    for v in range(V)]
        new = [[z - lr * gz for z, lr, gz in zip(acts[v], lrs, g[v])]
               for v in range(V)]
        new = [SP.prox_zs(new[v], cfg) for v in range(V)]          # case B
        if a is None:
            return new, None
        # case C: Jacobi -- the code steps at the PRE-step activities
        return new, [SP.a_step(params, acts[v], a[v], cfg, eta_g, x=Zin[v]) for v in range(V)]

    def dual(acts, duals):
        if all(al == 0.0 for al in alphas):
            return duals
        r = [residuals(params, acts[v], Zin[v], cfg.enc, norm) for v in range(V)]
        return [[lam + al * ri for lam, al, ri in zip(duals[v], alphas, r[v])]
                for v in range(V)]

    if not sparse_on:
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

        acts = primal(acts, duals)                   # Algorithm 1 line 10
        a = None
        F_last = F_total(acts, duals)
    else:
        def body_sp(carry, _):
            acts, duals, a = carry
            acts, a = primal(acts, duals, a)
            duals = dual(acts, duals)
            return (acts, duals, a), (F_report(acts, duals, a), loss_z(out_stack(acts)))

        if T > 1:
            (acts, duals, a), (F_h, L_h) = jax.lax.scan(
                body_sp, (acts0, duals0, a0), xs=None, length=T - 1)
        else:
            acts, duals, a, F_h, L_h = acts0, duals0, a0, None, None

        acts, a = primal(acts, duals, a)             # Algorithm 1 line 10
        F_last = F_report(acts, duals, a)
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
    hist = {"F": F_hist, "L": L_hist}
    if sparse_on:
        hist["F0"] = F_report(acts0, duals0, a0)
        # pilot stability gate: the largest activity norm at the end, relative to
        # the largest activity norm at the start (per view).  NOT F_T / F_0 -- the
        # completed-square energy grows with the multipliers by design -- and NOT a
        # per-layer ratio: layers at and above s can START near zero (thresholded
        # code; s is non-residual), which made a per-layer ratio read 505x with
        # nothing unstable.  A genuine blow-up anywhere exceeds the start scale.
        hist["act_growth"] = jnp.max(jnp.stack([
            jnp.max(jnp.stack([jnp.linalg.norm(z) for z in acts[v]]))
            / (jnp.max(jnp.stack([jnp.linalg.norm(z) for z in acts0[v]])) + 1e-12)
            for v in range(V)]))
        if a is not None:
            hist["a"] = a
    return acts, duals_w, hist


# Weight gradient from relaxed primal-dual state.
# Activities and duals held fixed (stop_gradient); differentiate augmented
# Lagrangian wrt weights only (same pattern as pc_grads / pc-alm method_grad).
# Update is local: each weight sees only its layer's composite credit
# rho * r_l + lam_l and the activity below it.
def grads(params, Zin, A, cfg: RunCfg, key):
    import equinox as eqx

    acts, duals, hist = relax(params, Zin, A, cfg, key)
    acts = jax.lax.stop_gradient(acts)
    duals = jax.lax.stop_gradient(duals)
    V = Zin.shape[0]
    a = jax.lax.stop_gradient(hist["a"]) if "a" in hist else None

    def F_of_params(p):
        F = sum(al_energy(p, acts[v], Zin[v], duals[v], cfg.pc.rho, cfg.enc,
                          cfg.pc.out_scale, SP.norm_spec(cfg)) for v in range(V))
        if a is not None:            # case C: D's gradient from the head energy
            F = F + sum(SP.head_smooth(p, acts[v], a[v], cfg, x=Zin[v]) for v in range(V))
        return F

    g = eqx.filter_grad(F_of_params)(params)
    aux = {"F_final": hist["F"][-1], "L_final": hist["L"][-1],
           "F_hist": hist["F"], "L_hist": hist["L"],
           "dual_norm": jnp.sqrt(sum(jnp.sum(l ** 2)
                                     for dv in duals for l in dv))}
    if SP.active(cfg):
        aux["F_first"] = hist["F0"]
        aux["act_growth"] = hist["act_growth"]
        if SP.is_D(cfg):
            # relaxation convergence at the head layers: mean ||r_l|| / ||f_l|| over
            # heads and views (review D-M7; z_l -> f_l is what makes the deployed
            # code the counterpart of the relaxed one)
            rr = []
            for v in range(V):
                pr = layer_predictions(params, acts[v], Zin[v], cfg.enc)
                rr += [jnp.linalg.norm(acts[v][l] - pr[l]) / (jnp.linalg.norm(pr[l]) + 1e-12)
                       for l in cfg.sparse.heads]
            aux["head_resid"] = sum(rr) / len(rr)
        aux["active_frac"] = sum(SP.active_fraction(acts[v], None if a is None else a[v], cfg)
                                 for v in range(V)) / V
    return g, aux
