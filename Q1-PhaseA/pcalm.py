# PC-ALM (predictive-coding augmented Lagrangian method): primal-dual activity relaxation and the resulting
# weight gradient for the encoder, matching jpc.pc_energy_fn term-for-term when duals are zero (rho=1).
# Supports the sparse-coding arms (cases B/C/D) via sparse.py hooks.
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


def layer_predictions(params, acts: List[jnp.ndarray], x, enc: EncoderCfg, norm=None):
    """Returns the per-layer prediction f_l(z_{l-1}) for every layer, in the same order as acts (z_0 = x excluded).
    Mirrors jpc.pc_energy_fn's term structure term-for-term."""
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


def residuals(params, acts, x, enc: EncoderCfg, norm=None):
    """Returns r_l = z_l - f_l(z_{l-1}) for every layer; norm=(s, c) normalises the prediction at layer s."""
    return [a - p for a, p in zip(acts, layer_predictions(params, acts, x, enc, norm))]


def zero_duals(acts):
    """Returns zero-initialized duals, one per activity per view (Algorithm 1 line 5)."""
    return [[jnp.zeros_like(z) for z in acts_v] for acts_v in acts]


# ------------------------------------------------------------------ energy


def al_energy(params, acts, x, duals, rho: float, enc: EncoderCfg,
              out_scale: Optional[float] = None, norm=None):
    """Computes the augmented Lagrangian L_rho = (1/2B) rho sum_l ||r_l + lam_l/rho||^2 (paper Eq. 9).
    duals=None gives plain PC energy; matches jpc.pc_energy_fn exactly at rho=1 with duals=None."""
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


def _objective(cfg: RunCfg, A):
    """Returns a closure computing the LeJEPA loss on a stack of per-view embeddings, with cfg and A bound in."""
    def loss_z(Zv):
        """Computes the LeJEPA loss on embedding stack Zv."""
        return S.lejepa_loss(
            Zv, A, cfg.data.n_global_views, cfg.sig.lambd, cfg.use_sigreg,
            projector=cfg.projector, t_max=cfg.sig.t_max,
            n_points=cfg.sig.n_points, scale_by_n=cfg.sig.scale_by_n)[0]
    return loss_z


def relax(params, Zin, A, cfg: RunCfg, key):
    """Runs Algorithm 1's primal-dual relaxation (T-1 primal/dual cycles, then a final primal step).
    alpha=0 keeps duals at zero (exactly the old PC relaxation); returns (acts, duals, history)."""
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
        """Stacks the final (output) activity of every view into one array."""
        return jnp.stack([acts[v][-1] for v in range(V)])

    def F_total(acts, duals):
        """Sums the augmented Lagrangian over all views."""
        return sum(al_energy(params, acts[v], Zin[v], duals[v], rho, cfg.enc,
                             cfg.pc.out_scale, norm) for v in range(V))

    def F_smooth(acts, duals, a):
        """Returns the differentiable part of the total energy, adding the sparse-arm head terms if active."""
        if not sparse_on:
            return F_total(acts, duals)
        return F_total(acts, duals) + sum(
            SP.head_smooth(params, acts[v], None if a is None else a[v], cfg, x=Zin[v])
            for v in range(V))

    def F_report(acts, duals, a):
        """Returns the full reported energy, adding the sparse-arm non-smooth (L1) terms to F_smooth."""
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
        """Performs one gradient-descent step on the activities (and sparse code a, if active)."""
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
        """Performs one dual-ascent step on the multipliers, lam += alpha * r, per layer and view."""
        if all(al == 0.0 for al in alphas):
            return duals
        r = [residuals(params, acts[v], Zin[v], cfg.enc, norm) for v in range(V)]
        return [[lam + al * ri for lam, al, ri in zip(duals[v], alphas, r[v])]
                for v in range(V)]

    if not sparse_on:
        def body(carry, _):
            """Performs one primal step then one dual step; scanned over the relaxation steps."""
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
            """Performs one primal step (with sparse code a) then one dual step; scanned over the steps."""
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
        # pilot stability gate: largest activity norm at the end vs. the start (per view), NOT F_T / F_0
        # (grows with the multipliers by design) and NOT per-layer (layers >= s start near zero, which
        # read as a false 505x). A genuine blow-up anywhere exceeds the start scale.
        hist["act_growth"] = jnp.max(jnp.stack([
            jnp.max(jnp.stack([jnp.linalg.norm(z) for z in acts[v]]))
            / (jnp.max(jnp.stack([jnp.linalg.norm(z) for z in acts0[v]])) + 1e-12)
            for v in range(V)]))
        if a is not None:
            hist["a"] = a
    return acts, duals_w, hist


def grads(params, Zin, A, cfg: RunCfg, key):
    """Relaxes the activities/duals, then returns the weight gradient of the augmented Lagrangian at that
    fixed state (stop-gradiented), plus a dict of diagnostics."""
    import equinox as eqx

    acts, duals, hist = relax(params, Zin, A, cfg, key)
    acts = jax.lax.stop_gradient(acts)
    duals = jax.lax.stop_gradient(duals)
    V = Zin.shape[0]
    a = jax.lax.stop_gradient(hist["a"]) if "a" in hist else None

    def F_of_params(p):
        """Computes the augmented Lagrangian as a function of weights p, at the fixed relaxed state."""
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
