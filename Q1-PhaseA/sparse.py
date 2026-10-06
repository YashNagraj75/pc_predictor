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


def soft_threshold(v, theta):
    """S_theta(v) = sign(v) max(|v| - theta, 0).  Derivative 1[|v| > theta]."""
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - theta, 0.0)


def normalise_dict(D, eps=1e-8):
    """Unit-norm columns."""
    return D / jnp.maximum(jnp.linalg.norm(D, axis=0, keepdims=True), eps)


def spectral_sq(D):
    """||D||_2^2, exact."""
    return jnp.linalg.norm(D, ord=2) ** 2


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
    if cfg.sparse.case != "C":
        return params
    return (params[0], params[1], normalise_dict(params[2]))


def get_D(params):
    return params[2]


def param_labels(params):
    """optax.multi_transform labels for case-C params: encoder vs dictionary."""
    enc = jax.tree_util.tree_map(lambda _: "enc", (params[0], params[1]))
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


def _pred(params, scalings, l, prev):
    """f_l(prev), mirroring jpc.init_activities_with_ffwd term for term."""
    model, skip = params[0], params[1]
    p = scalings[l] * vmap(model[l])(prev)
    if skip[l] is not None:
        p = p + vmap(skip[l])(prev)
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
    rho = cfg.pc.rho
    sc = _scalings(params, x, cfg)
    L, s = len(params[0]), sp.layer
    acts, extras, prev = [], {}, x
    for l in range(L):
        f = _pred(params, sc, l, prev)
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
                    # the step size is a constant of the solver, not a path for
                    # gradients (matters only for C2's backprop through ISTA)
                    eta = jax.lax.stop_gradient(sp.step_frac / (mu * spectral_sq(D)))
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
    """C2's head term: batch mean of mu/2||f_s - D a^(T)||^2 + gamma||a^(T)||_1."""
    if cfg.sparse.case != "C":
        return 0.0
    _, ex = deploy(params, x, cfg)
    return jnp.mean(lasso_energy(ex["f_s"], get_D(params), ex["a"],
                                 cfg.sparse.gamma, lasso_mu(cfg)))


# --------------------------------------------------- relaxation hooks

def layer_alphas(cfg, L):
    """Per-layer dual rates.  case none: alpha everywhere (unchanged)."""
    a = cfg.pc.alpha
    if not active(cfg):
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
    if cfg.sparse.case != "C":
        return 0.0
    zs = acts[cfg.sparse.layer]
    return 0.5 * cfg.sparse.beta_g * jnp.sum((zs - a @ get_D(params).T) ** 2) / zs.shape[0]


def nonsmooth(acts, a, cfg):
    """The L1 terms, batch mean.  Reported in F; never differentiated."""
    sp = cfg.sparse
    if sp.case == "B":
        zs = acts[sp.layer]
        return sp.beta * jnp.sum(jnp.abs(zs)) / zs.shape[0]
    if sp.case == "C":
        return sp.gamma * jnp.sum(jnp.abs(a)) / a.shape[0]
    return 0.0


def prox_zs(acts, cfg):
    """Case B proximal step on z_s, PER-SAMPLE threshold eta_h * beta."""
    if cfg.sparse.case != "B":
        return acts
    s = cfg.sparse.layer
    out = list(acts)
    out[s] = soft_threshold(acts[s], cfg.pc.eta_h * cfg.sparse.beta)
    return out


def a_step(params, acts, a, cfg):
    """Case C proximal-gradient step on the code, at the PRE-step activities.

    a <- S_{eta_g gamma}(a - eta_g beta_g D^T (D a - z_s)),
    eta_g = step_frac / (beta_g ||D||^2).  The gradient is per sample (no B).
    """
    sp = cfg.sparse
    D = get_D(params)
    zs = acts[sp.layer]
    eta = sp.step_frac / (sp.beta_g * spectral_sq(D))
    grad = sp.beta_g * ((a @ D.T - zs) @ D)
    return soft_threshold(a - eta * grad, eta * sp.gamma)


def active_fraction(acts, a, cfg):
    """Fraction of nonzero units in the sparse variable (z_s for B, a for C)."""
    sp = cfg.sparse
    v = acts[sp.layer] if sp.case == "B" else a
    return jnp.mean((v != 0).astype(jnp.float32))


# ------------------------------------------------------------- health

def health(params, x, cfg):
    """Scalars for the training log (rescaling cheat, sparsity, dictionary)."""
    if not active(cfg):
        return {}
    sp = cfg.sparse
    acts, ex = deploy(params, x, cfg)
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
