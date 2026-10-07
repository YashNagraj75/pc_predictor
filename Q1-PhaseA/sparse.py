"""Sparse prior on one layer of the PC-ALM encoder: cases B and C.

Spec (the contract this file is reviewed against):
    Obsidian vault, PCWM/Methods/'Method - Sparse PC-ALM arms spec.md'
Derivation and motivation:
    PCWM/Questions/'Question - How to add sparsity to PC-ALM.md'

    case "none" : the original encoder.  Every function here then delegates to
                  encoder.py / leaves params untouched, so the old code path is
                  reproduced bit for bit (pinned by test_sparse.py against the
                  pre-edit golden_pcalm_alpha*.npz).
    case "B"    : L1 prior beta*||z_s||_1 on layer s, multiplier at s removed.
                  Deployed:  z_s = S_{beta/rho}(f_s(z_{s-1})).
    case "C"    : multiplier at s removed + top-down dictionary head
                  (beta_g/2)||z_s - D a||^2 + gamma*||a||_1.
                  Deployed:  t ISTA steps on the LASSO
                      min_a  mu/2 ||f_s - D a||^2 + gamma ||a||_1,
                      mu = rho*beta_g/(rho+beta_g),
                  then  z_s = (rho f_s + beta_g D a) / (rho + beta_g).

Shared architecture change for B and C: layer s is NON-residual (skip[s] = None)
and W_s is multiplied by sqrt(L) at initialisation.  jpc keeps the residual
scaling 1/sqrt(N L) on every interior layer unless ALL skips are removed, so
without the rescale z_s would be ~sqrt(L) smaller than the residual stream.

Conventions
-----------
* Activations are row-major: a batch of codes is (B, m), the dictionary is
  (n, m), so  D a  for a batch is  a @ D.T  and  D^T f  is  f @ D.
* Energies follow jpc's BATCH-MEAN convention (sum over samples / B), so the
  relaxation's activity step on them is eta_h * B.  Every proximal threshold
  and every analytically computed per-sample gradient here uses the per-sample
  step eta_h -- never eta_h * B.
* ||D||_2 is the exact spectral norm (jnp.linalg.norm(D, 2)), not a power
  iteration: power iteration under-estimates, which would push the ISTA step
  past 1/Lipschitz.
"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jpc
from jax import vmap

import encoder as E


# ------------------------------------------------------------------ basics

def active(cfg) -> bool:
    return cfg.sparse.case != "none"


def is_D(cfg) -> bool:
    """Case D: side dictionary heads on the residual stream, all multipliers kept."""
    return cfg.sparse.case == "D"


def check_D(cfg, L):
    sp = cfg.sparse
    assert len(sp.heads) >= 1, "case D needs at least one head"
    assert len(sp.head_ws) == len(sp.heads) == len(sp.gammas), \
        f"heads/head_ws/gammas lengths differ: {sp.heads} {sp.head_ws} {sp.gammas}"
    assert len(set(sp.heads)) == len(sp.heads), f"duplicate heads {sp.heads}"
    assert all(0 < l < L - 1 for l in sp.heads), f"heads {sp.heads} must be interior (1..{L-2})"


def soft_threshold(v, theta):
    """S_theta(v) = sign(v) max(|v| - theta, 0).  Derivative 1[|v| > theta]."""
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - theta, 0.0)


def normalise_dict(D, eps=1e-8):
    """Unit-norm columns."""
    return D / jnp.maximum(jnp.linalg.norm(D, axis=0, keepdims=True), eps)


def spectral_sq(D):
    """||D||_2^2, exact.  The step size is a solver constant, never a gradient
    path: stop_gradient is applied to D itself, so no SVD derivative is built
    (stopping only the OUTPUT still built the JVP -- found in review)."""
    return jnp.linalg.norm(jax.lax.stop_gradient(D), ord=2) ** 2


def code_step(params, cfg):
    """Case-C relaxation code step eta_g = step_frac / (beta_g ||D||^2).
    D is constant within one relaxation, so this is computed ONCE per relax.
    Case D: one step per head (tuple).  w_l cancels from the code dynamics:
    the code energy is w_l [beta_g/2||z_l - D_l a||^2 + gamma_l||a||_1], whose
    prox-gradient step at eta/w_l is the unweighted step at eta."""
    if is_D(cfg):
        return tuple(cfg.sparse.step_frac / (cfg.sparse.beta_g * spectral_sq(Dl))
                     for Dl in get_D(params))
    return cfg.sparse.step_frac / (cfg.sparse.beta_g * spectral_sq(get_D(params)))


def huber_energy(f, beta, rho):
    """Per-sample Moreau envelope of beta||.||_1 at f:
    min_z rho/2||z - f||^2 + beta||z||_1 = rho/2||f - S(f)||^2 + beta||S(f)||_1,
    S = S_{beta/rho}.  -> (B,)"""
    zs = soft_threshold(f, beta / rho)
    return 0.5 * rho * jnp.sum((f - zs) ** 2, -1) + beta * jnp.sum(jnp.abs(zs), -1)


def lasso_mu(cfg):
    rho, bg = cfg.pc.rho, cfg.sparse.beta_g
    return rho * bg / (rho + bg)


# ------------------------------------------------------------------ params

def init_params(key, cfg, input_dim):
    """E.init, plus the case-B/C architecture change and (case C) D.

    The encoder weights use exactly the key E.init uses, so the shared layers
    of a sparse arm start from the same draw as the original encoder.  D comes
    from an independent fold of the same key.
    """
    params = E.init(key, cfg.enc, input_dim)
    sp = cfg.sparse
    if sp.case == "none":
        return params
    if sp.case == "D":
        # encoder UNCHANGED (same draw as case none); one dictionary per head,
        # each from its own fold so D_l is identical across nested head sets
        check_D(cfg, len(params[0]))
        n, m = cfg.enc.width, sp.dict_mult * cfg.enc.width
        Ds = tuple(normalise_dict(jax.random.normal(jax.random.fold_in(key, 7919 + l), (n, m)))
                   for l in sp.heads)
        return (params[0], params[1], Ds)
    model, skip = params
    L, s = len(model), sp.layer
    assert 0 < s < L - 1, f"sparse.layer={s} must be interior (1..{L-2})"
    skip = list(skip)
    assert skip[s] is not None, "layer s is already non-residual"
    skip[s] = None
    model = eqx.tree_at(lambda m: m[s].layers[1].weight, model,
                        model[s].layers[1].weight * math.sqrt(L))
    if sp.case == "B":
        return (model, skip)
    n, m = cfg.enc.width, sp.dict_mult * cfg.enc.width
    D = jax.random.normal(jax.random.fold_in(key, 7919), (n, m))
    return (model, skip, normalise_dict(D))


def project(params, cfg):
    """Post-optimiser-step projection: re-normalise D's columns (case C)."""
    if cfg.sparse.case == "D":
        return (params[0], params[1], tuple(normalise_dict(Dl) for Dl in params[2]))
    if cfg.sparse.case != "C":
        return params
    return (params[0], params[1], normalise_dict(params[2]))


def get_D(params):
    return params[2]


def param_labels(params):
    """optax.multi_transform labels for case-C params: encoder vs dictionary."""
    enc = jax.tree_util.tree_map(lambda _: "enc", (params[0], params[1]))
    if isinstance(params[2], tuple):                       # case D: one D per head
        return (enc[0], enc[1], tuple("dict" for _ in params[2]))
    return (enc[0], enc[1], "dict")


# ----------------------------------------------------------------- LASSO

def ista(f, D, gamma, mu, t, eta, a0=None):
    """t ISTA steps on  mu/2||f - D a||^2 + gamma||a||_1, rows of f independent.

    a0=None starts from a^(0) = S_{eta gamma}(eta mu D^T f), which is one ISTA
    step from a = 0 (the untrained one-stage LISTA initialiser).
    """
    th = eta * gamma
    if a0 is None:
        a0 = soft_threshold(eta * mu * (f @ D), th)

    def step(_, a):
        return soft_threshold(a + eta * mu * ((f - a @ D.T) @ D), th)

    return jax.lax.fori_loop(0, t, step, a0) if t > 0 else a0


def fista(f, D, gamma, mu, n_iter=2000):
    """Reference LASSO solver (FISTA, exact step 1/(mu ||D||^2))."""
    eta = 1.0 / (mu * spectral_sq(D))
    th = eta * gamma

    def step(_, c):
        a, y, tk = c
        a_new = soft_threshold(y + eta * mu * ((f - y @ D.T) @ D), th)
        tk1 = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * tk ** 2))
        return a_new, a_new + ((tk - 1.0) / tk1) * (a_new - a), tk1

    a0 = jnp.zeros((f.shape[0], D.shape[1]), f.dtype)
    a, _, _ = jax.lax.fori_loop(0, n_iter, step, (a0, a0, jnp.asarray(1.0, f.dtype)))
    return a


def lasso_energy(f, D, a, gamma, mu):
    """Per-sample  mu/2||f - D a||^2 + gamma||a||_1  -> (B,)."""
    return 0.5 * mu * jnp.sum((f - a @ D.T) ** 2, -1) + gamma * jnp.sum(jnp.abs(a), -1)


def kkt_violation(f, D, a, gamma, mu):
    """Per-sample max violation of the LASSO optimality conditions, / gamma.

    With c = mu D^T (f - D a):  a_j != 0 needs c_j = gamma sign(a_j);
    a_j == 0 needs |c_j| <= gamma.
    """
    c = mu * ((f - a @ D.T) @ D)
    on = jnp.abs(c - gamma * jnp.sign(a))
    off = jnp.maximum(jnp.abs(c) - gamma, 0.0)
    v = jnp.where(a != 0, on, off)
    return jnp.max(v, -1) / jnp.maximum(gamma, 1e-12)


# ------------------------------------------------------- deployed forward

def _scalings(params, x, cfg):
    model, skip = params[0], params[1]
    return jpc._get_param_scalings(model=model, input=x, skip_model=skip,
                                   param_type=cfg.enc.parameterisation,
                                   gamma=E.gamma_for(cfg.enc))


def norm_spec(cfg):
    """(s, c) if the sparse layer's prediction is normalised, else None."""
    sp = cfg.sparse
    if sp.case in ("none", "D") or not sp.normalise_fs:
        return None
    return (sp.layer, sp.fs_norm)


NORM_EPS = 1e-6


def normalise_rows(f, c):
    """c f / sqrt(||f||^2 + eps^2), per row.  The SAFE norm matters: an
    all-black augmented view gives f_s = 0 exactly (the encoder has no biases),
    and the gradient of f / ||f|| there is 0/0 -- that NaN stopped all four
    pilot runs at step 68 of job b3c717c6 (one all-zero view in that batch)."""
    return c * f / jnp.sqrt(jnp.sum(f * f, axis=-1, keepdims=True) + NORM_EPS ** 2)


def _pred(params, scalings, l, prev, norm=None):
    """f_l(prev), mirroring jpc.init_activities_with_ffwd term for term;
    norm = (s, c) normalises layer s's prediction (as pcalm.layer_predictions)."""
    model, skip = params[0], params[1]
    p = scalings[l] * vmap(model[l])(prev)
    if skip[l] is not None:
        p = p + vmap(skip[l])(prev)
    if norm is not None and l == norm[0]:
        p = normalise_rows(p, norm[1])
    return p


def deploy(params, x, cfg, t=None, beta_g=None):
    """The deployed (test-time) network.  -> (acts, extras).

    acts   : [z_0, ..., z_{L-1}] in encoder.forward's order (z_{L-1} = embedding).
    extras : {} for case none; {"f_s"} for B; {"f_s", "a", "eta"} for C.

    t      : case-C ISTA steps (default cfg.sparse.T_test).  t = 0 uses a^(0).
    beta_g : case-C test-time coupling override (the beta_g dial).  beta_g = 0
             returns exactly the plain forward pass of the modified network.
    """
    sp = cfg.sparse
    if sp.case == "none":
        z, hs = E.forward(params, x, cfg.enc)
        return [*hs, z], {}
    if sp.case == "D":
        # main path is the ORIGINAL encoder; the heads only read it.  With all
        # multipliers kept, z_l is pinned to f_l at the fixed point, so each
        # code solves  min_a beta_g/2||f_l - D_l a||^2 + gamma_l||a||_1
        # (mu = beta_g, and w_l does not change the minimiser).
        z, hs = E.forward(params, x, cfg.enc)
        acts = [*hs, z]
        tt = sp.T_test if t is None else t
        codes, steps = [], []
        for l, Dl, gl in zip(sp.heads, get_D(params), sp.gammas):
            eta = sp.step_frac / (sp.beta_g * spectral_sq(Dl))
            codes.append(ista(acts[l], Dl, gl, sp.beta_g, tt, eta))
            steps.append(eta)
        return acts, {"a": tuple(codes), "eta": tuple(steps),
                      "f_heads": tuple(acts[l] for l in sp.heads)}
    rho = cfg.pc.rho
    sc = _scalings(params, x, cfg)
    L, s = len(params[0]), sp.layer
    acts, extras, prev = [], {}, x
    for l in range(L):
        f = _pred(params, sc, l, prev, norm_spec(cfg))
        if l == s:
            extras["f_s"] = f
            if sp.case == "B":
                f = soft_threshold(f, sp.beta / rho)
            else:
                bg = sp.beta_g if beta_g is None else beta_g
                tt = sp.T_test if t is None else t
                if bg == 0:
                    extras["a"] = None
                else:
                    D = get_D(params)
                    mu = rho * bg / (rho + bg)
                    # solver constant; spectral_sq stops the gradient at D
                    eta = sp.step_frac / (mu * spectral_sq(D))
                    a = ista(f, D, sp.gamma, mu, tt, eta)
                    extras["a"], extras["eta"] = a, eta
                    f = (rho * f + bg * (a @ D.T)) / (rho + bg)
        acts.append(f)
        prev = f
    return acts, extras


def forward(params, x, cfg, t=None):
    """encoder.forward's (z, hs) contract for the deployed network."""
    acts, _ = deploy(params, x, cfg, t=t)
    return acts[-1], acts[:-1]


def deployed_head_energy(params, x, cfg):
    """The deployed energy of the sparse layer, batch mean, for ONE view.

    The energy the PC arm minimises over z_s, at the deployed z_s:
      B (C1-B): Huber_{beta,rho}(f_s), the Moreau envelope of beta||.||_1
      C (C2)  : mu/2||f_s - D a^(T)||^2 + gamma||a^(T)||_1
    train.bp_objective SUMS this over views, as the PC energy does."""
    sp = cfg.sparse
    if sp.case == "none":
        return 0.0
    _, ex = deploy(params, x, cfg)
    if sp.case == "D":
        # BP-D: sum_l w_l [beta_g/2||f_l - D_l a_l^(T)||^2 + gamma_l||a_l^(T)||_1]
        return sum(wl * jnp.mean(lasso_energy(f, Dl, a, gl, sp.beta_g))
                   for f, Dl, a, gl, wl in zip(ex["f_heads"], get_D(params), ex["a"],
                                               sp.gammas, sp.head_ws))
    if sp.case == "B":
        return jnp.mean(huber_energy(ex["f_s"], sp.beta, cfg.pc.rho))
    return jnp.mean(lasso_energy(ex["f_s"], get_D(params), ex["a"], sp.gamma, lasso_mu(cfg)))


# --------------------------------------------------- relaxation hooks

def layer_alphas(cfg, L):
    """Per-layer dual rates.  case none: alpha everywhere (unchanged)."""
    a = cfg.pc.alpha
    if not active(cfg) or is_D(cfg):          # case D keeps every multiplier
        return [a] * L
    s = cfg.sparse.layer
    return [0.0 if (l == s or (not cfg.sparse.below_hard and l < s)) else a
            for l in range(L)]


def relax_init(params, x, cfg):
    """Relaxation start for ONE view: the deployed forward pass (C at t = 0).
    -> (acts, a or None)."""
    acts, ex = deploy(params, x, cfg, t=0)
    return acts, ex.get("a")


def head_smooth(params, acts, a, cfg):
    """Smooth head energy (case C), batch mean: beta_g/(2B) sum ||z_s - D a||^2."""
    if is_D(cfg):
        sp = cfg.sparse
        return sum(wl * 0.5 * sp.beta_g * jnp.sum((acts[l] - al @ Dl.T) ** 2) / acts[l].shape[0]
                   for l, Dl, al, wl in zip(sp.heads, get_D(params), a, sp.head_ws))
    if cfg.sparse.case != "C":
        return 0.0
    zs = acts[cfg.sparse.layer]
    return 0.5 * cfg.sparse.beta_g * jnp.sum((zs - a @ get_D(params).T) ** 2) / zs.shape[0]


def nonsmooth(acts, a, cfg):
    """The L1 terms, batch mean.  Reported in F; never differentiated."""
    sp = cfg.sparse
    if sp.case == "D":
        return sum(wl * gl * jnp.sum(jnp.abs(al)) / al.shape[0]
                   for al, gl, wl in zip(a, sp.gammas, sp.head_ws))
    if sp.case == "B":
        zs = acts[sp.layer]
        return sp.beta * jnp.sum(jnp.abs(zs)) / zs.shape[0]
    if sp.case == "C":
        return sp.gamma * jnp.sum(jnp.abs(a)) / a.shape[0]
    return 0.0


def prox_zs(acts, cfg):
    """Case B proximal step on z_s.  PER-SAMPLE threshold = (per-sample step
    at layer s) * beta = (s_step / rho) * beta -- layer s has its own step
    (pcalm.relax), never eta_h * B."""
    if cfg.sparse.case != "B":
        return acts
    s = cfg.sparse.layer
    out = list(acts)
    out[s] = soft_threshold(acts[s], cfg.sparse.s_step / cfg.pc.rho * cfg.sparse.beta)
    return out


def a_step(params, acts, a, cfg, eta):
    """Case C proximal-gradient step on the code, at the PRE-step activities.

    a <- S_{eta_g gamma}(a - eta_g beta_g D^T (D a - z_s)),
    eta = eta_g from code_step(), computed once per relaxation.  The gradient
    is per sample (no B).
    """
    sp = cfg.sparse
    if is_D(cfg):           # one prox-gradient step per head; eta is a tuple
        return tuple(soft_threshold(al - el * sp.beta_g * ((al @ Dl.T - acts[l]) @ Dl), el * gl)
                     for l, Dl, al, el, gl in zip(sp.heads, get_D(params), a, eta, sp.gammas))
    D = get_D(params)
    zs = acts[sp.layer]
    grad = sp.beta_g * ((a @ D.T - zs) @ D)
    return soft_threshold(a - eta * grad, eta * sp.gamma)


def active_fraction(acts, a, cfg):
    """Fraction of nonzero units in the sparse variable (z_s for B, a for C)."""
    sp = cfg.sparse
    if is_D(cfg):           # mean over heads
        return sum(jnp.mean((al != 0).astype(jnp.float32)) for al in a) / len(a)
    v = acts[sp.layer] if sp.case == "B" else a
    return jnp.mean((v != 0).astype(jnp.float32))


# ------------------------------------------------------------- health

def health(params, x, cfg):
    """Scalars for the training log (rescaling cheat, sparsity, dictionary)."""
    if not active(cfg):
        return {}
    sp = cfg.sparse
    acts, ex = deploy(params, x, cfg)
    if is_D(cfg):
        return health_D(params, acts, ex, cfg)
    s = sp.layer
    model = params[0]
    out = {
        "zs_rownorm": float(jnp.mean(jnp.linalg.norm(acts[s], axis=-1))),
        "fs_rownorm": float(jnp.mean(jnp.linalg.norm(ex["f_s"], axis=-1))),
        "Ws_fro": float(jnp.linalg.norm(model[s].layers[1].weight)),
        "Wnext_fro": float(jnp.linalg.norm(model[s + 1].layers[1].weight)),
        "active_frac": float(active_fraction(acts, ex.get("a"), cfg)),
    }
    if sp.case == "C":
        D, a = get_D(params), ex["a"]
        G = jnp.abs(D.T @ D) - jnp.eye(D.shape[1])
        out.update({
            "coherence": float(jnp.max(G)),
            "dead_atoms_frac": float(jnp.mean(jnp.all(a == 0, axis=0))),
            "D_spectral_sq": float(spectral_sq(D)),
            "lasso_energy": float(jnp.mean(lasso_energy(ex["f_s"], D, a, sp.gamma,
                                                        lasso_mu(cfg)))),
        })
    return out


def _spread(v):
    """1 - median cosine to the mean direction (0 = every row points the same way)."""
    m = jnp.mean(v, 0)
    c = (v @ m) / (jnp.linalg.norm(v, axis=1) * jnp.linalg.norm(m) + 1e-12)
    return float(1.0 - jnp.median(c))


def health_D(params, acts, ex, cfg):
    """Case D log scalars: per head (sparsity, supports, dictionary) and the
    collapse guards of the pilot gate (supports, embedding spread)."""
    import numpy as np
    sp = cfg.sparse
    z = acts[-1]
    out = {"embed_spread": _spread(z), "embed_norm": float(jnp.mean(jnp.linalg.norm(z, axis=-1))),
           "embed_std": float(jnp.mean(jnp.std(z, 0))), "n_rows": int(z.shape[0])}
    fr, sup = [], []
    for l, Dl, al, gl in zip(sp.heads, get_D(params), ex["a"], sp.gammas):
        on = np.asarray(al != 0)
        f = acts[l]
        G = jnp.abs(Dl.T @ Dl) - jnp.eye(Dl.shape[1])
        h = {f"h{l}_active": float(on.mean()), f"h{l}_supports": len({r.tobytes() for r in on}),
             f"h{l}_dead_atoms": float(np.mean(~on.any(0))), f"h{l}_coherence": float(jnp.max(G)),
             f"h{l}_lasso": float(jnp.mean(lasso_energy(f, Dl, al, gl, sp.beta_g))),
             f"h{l}_rel_recon": float(jnp.mean(jnp.linalg.norm(f - al @ Dl.T, axis=-1)
                                               / (jnp.linalg.norm(f, axis=-1) + 1e-12))),
             f"h{l}_input_spread": _spread(f)}
        out.update(h)
        fr.append(h[f"h{l}_active"]); sup.append(h[f"h{l}_supports"])
    out["active_frac"] = float(np.mean(fr))
    out["active_min"], out["active_max"] = float(min(fr)), float(max(fr))
    out["supports_min"] = int(min(sup))
    return out
