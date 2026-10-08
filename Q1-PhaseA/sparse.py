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
import numpy as np
import jpc
from jax import vmap

import encoder as E


# ------------------------------------------------------------------ basics

def active(cfg) -> bool:
    return cfg.sparse.case != "none"


def is_D(cfg) -> bool:
    """Case D: side dictionary heads on the residual stream, all multipliers kept."""
    return cfg.sparse.case == "D"


def head_in(v, mu):
    """What a case-D head explains: the CENTRED, per-sample normalised activity
        sqrt(n) (v - mu) / ||v - mu||       (safe norm, mu held fixed: no gradient)
    with mu the running mean of the head layer (params[3], EMA, never trained).

    History.  Raw v: the heads shrank the stream (||f_5|| 2.35 -> 1.36).  Normalised
    v (2026-10-07): closed that, but the 2000-step pilot (job 4faac68e) found a
    constant-offset route -- both arms grew the head layers ~6-12x toward ONE
    input-independent direction that a single code explains.  Centring removes the
    shared offset, so the head must explain how inputs DIFFER (user decision
    2026-10-08).  A remaining route -- variation squeezed into few directions -- is
    not removed; health_D logs the participation ratio and the gate guards it."""
    return normalise_rows(v - jax.lax.stop_gradient(mu), float(np.sqrt(v.shape[-1])))


def get_mu(params):
    return params[3]


def recon(cfg) -> bool:
    """Case E: case D plus a code -> pixel reconstruction anchor (beta_r > 0).
    Spec: wiki 'Method - Reconstruction-anchored sparse heads plan'.  Params are
    then (model, skip, Ds, mus, Gs, xbar); with beta_r = 0 they stay case D's 4-tuple
    and every expression below reduces to case D's exactly."""
    return is_D(cfg) and cfg.sparse.beta_r > 0


def get_G(params):
    return params[4]


def get_xbar(params):
    return params[5]


def stacked(f_t, Dl, xc, Gl, cfg):
    """The per-head LASSO as (target, dictionary, mu):
        case D:  (f~, D, beta_g)                       -> beta_g/2||f~ - D a||^2
        case E:  ([sqrt(bg) f~ ; sqrt(br) xc], [sqrt(bg) D ; sqrt(br) G], 1)
                 -> beta_g/2||f~ - D a||^2 + beta_r/2||xc - G a||^2
    so ista / fista / lasso_energy solve and score the anchored code unchanged."""
    sp = cfg.sparse
    if not recon(cfg):
        return f_t, Dl, sp.beta_g
    return (jnp.concatenate([jnp.sqrt(sp.beta_g) * f_t, jnp.sqrt(sp.beta_r) * xc], -1),
            stacked_D(Dl, Gl, cfg), 1.0)


def stacked_D(Dl, Gl, cfg):
    """[sqrt(beta_g) D ; sqrt(beta_r) G]  (case E's per-head LASSO dictionary)."""
    return jnp.concatenate([jnp.sqrt(cfg.sparse.beta_g) * Dl, jnp.sqrt(cfg.sparse.beta_r) * Gl], 0)


def _heads(params, cfg, x=None):
    """Per head: (layer, D_l, mu_l, G_l or None, centred pixels or None)."""
    sp = cfg.sparse
    Ds, mus = get_D(params), get_mu(params)
    if recon(cfg):
        assert x is not None, "case E needs the input view"
        xc = anchor_target(x, get_xbar(params))
        return [(l, Dl, mu, Gl, xc) for l, Dl, mu, Gl in zip(sp.heads, Ds, mus, get_G(params))]
    return [(l, Dl, mu, None, None) for l, Dl, mu in zip(sp.heads, Ds, mus)]


def mu_rows(Xtr, cfg, n=1024):
    """Calibration rows: initialise mu (and case E's xbar), and select gamma / beta_r.
    Same images, rng(0).  Case D: deterministic training centre views.  Case E (user
    decision 2026-10-08, review E-M1/D5): AUGMENTED training views, the distribution
    the model trains on -- one view per image from a fixed key (PRNGKey(0))."""
    import data as Dd
    idx = np.random.default_rng(0).choice(len(Xtr), n, replace=False)
    if recon(cfg):
        views, _ = Dd.make_views(jax.random.PRNGKey(0), jnp.asarray(Xtr[idx]), cfg.data)
        return views[0]
    return Dd.eval_batch(jnp.asarray(Xtr[idx]), cfg.data)


def t_deploy(cfg):
    """Deployed ISTA steps after a^(0): T_test * code_k, the count the relaxation takes."""
    return cfg.sparse.T_test * cfg.sparse.code_k


def anchor_target(x, xbar):
    """Case E target: the view minus the pixel mean, with each view's own mean
    brightness removed (user decision 2026-10-08) so brightness jitter is not rewarded."""
    xc = x - jax.lax.stop_gradient(xbar)
    return xc - jnp.mean(xc, axis=-1, keepdims=True)


def _head_means(params, x, cfg):
    z, hs = E.forward(params, x, cfg.enc)
    acts = [*hs, z]
    return tuple(jnp.mean(acts[l], axis=0) for l in cfg.sparse.heads)


def init_mu(params, x, cfg):
    """Set mu_l to the feedforward mean of head layer l over x (call before training,
    before head selection, and for every untrained floor)."""
    if not is_D(cfg):
        return params
    out = list(params)
    out[3] = _head_means(params, x, cfg)
    if recon(cfg):
        out[5] = jnp.mean(x, axis=0)          # per-pixel training mean, fixed after this
    return tuple(out)


@eqx.filter_jit
def _mu_ema(params, Zin, cfg):
    m = cfg.sparse.mu_momentum
    means = [jnp.mean(jnp.stack([_head_means(params, Zin[v], cfg)[h] for v in range(Zin.shape[0])]), axis=0)
             for h in range(len(cfg.sparse.heads))]
    return tuple(m * mu + (1 - m) * jax.lax.stop_gradient(nm) for mu, nm in zip(params[3], means))


def update_mu(params, Zin, cfg):
    """EMA update of mu from this step's training views (feedforward), after the
    optimiser step.  Same rule in both arms."""
    if not is_D(cfg):
        return params
    out = list(params)
    out[3] = _mu_ema(params, Zin, cfg)
    return tuple(out)


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
    Ds = jax.lax.stop_gradient(D)
    # exact: lambda_max(D D^T) = sigma_max(D)^2, via a symmetric eigensolve of the
    # n x n Gram, cheaper on GPU than the SVD behind norm(ord=2) (review D-M5/N7)
    # n x n Gram (or m x m when D is tall, e.g. case E's stacked [D; G]: same value)
    return jnp.linalg.eigvalsh(Ds @ Ds.T if Ds.shape[0] <= Ds.shape[1] else Ds.T @ Ds)[-1]


def code_step(params, cfg):
    """Case-C relaxation code step eta_g = step_frac / (beta_g ||D||^2).
    D is constant within one relaxation, so this is computed ONCE per relax.
    Case D: one step per head (tuple).  w_l cancels from the code dynamics:
    the code energy is w_l [beta_g/2||z_l - D_l a||^2 + gamma_l||a||_1], whose
    prox-gradient step at eta/w_l is the unweighted step at eta."""
    if recon(cfg):          # exact Lipschitz constant of the stacked LASSO
        sp = cfg.sparse
        return tuple(sp.step_frac / spectral_sq(stacked_D(Dl, Gl, cfg))
                     for Dl, Gl in zip(get_D(params), get_G(params)))
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
        mus = tuple(jnp.zeros((cfg.enc.width,)) for _ in sp.heads)   # set by init_mu
        if sp.beta_r > 0:      # case E: generative dictionary per head, own fold; pixel mean
            Gs = tuple(normalise_dict(jax.random.normal(jax.random.fold_in(key, 104729 + l), (input_dim, m)))
                       for l in sp.heads)
            return (params[0], params[1], Ds, mus, Gs, jnp.zeros((input_dim,)))   # xbar set by init_mu
        return (params[0], params[1], Ds, mus)
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
        out = list(params)
        out[2] = tuple(normalise_dict(Dl) for Dl in params[2])
        if len(params) == 6:                   # case E: unit-norm generative columns too
            out[4] = tuple(normalise_dict(Gl) for Gl in params[4])
        return tuple(out)
    if cfg.sparse.case != "C":
        return params
    return (params[0], params[1], normalise_dict(params[2]))


def get_D(params):
    return params[2]


def param_labels(params):
    """optax.multi_transform labels for case-C params: encoder vs dictionary."""
    enc = jax.tree_util.tree_map(lambda _: "enc", (params[0], params[1]))
    if isinstance(params[2], tuple):                       # case D: one D per head
        lab = (enc[0], enc[1], tuple("dict" for _ in params[2]),
               tuple("frozen" for _ in params[3]))         # mu: EMA only, never trained
        if len(params) == 6:                               # case E: G trained, xbar fixed
            lab = lab + (tuple("dict" for _ in params[4]), "frozen")
        return lab
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
        tt = t_deploy(cfg) if t is None else t
        codes, steps = [], []
        hd = _heads(params, cfg, x)
        fh = tuple(head_in(acts[l], mu) for l, _, mu, _, _ in hd)
        for f, (l, Dl, mu, Gl, xc), gl in zip(fh, hd, sp.gammas):
            F, Ds_, m_ = stacked(f, Dl, xc, Gl, cfg)
            eta = sp.step_frac / (m_ * spectral_sq(Ds_))
            codes.append(ista(F, Ds_, gl, m_, tt, eta))
            steps.append(eta)
        ex = {"a": tuple(codes), "eta": tuple(steps), "f_heads": fh}
        if recon(cfg):
            ex["x_c"] = hd[0][4]
        return acts, ex
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
        # BP-D: sum_l w_l [beta_g/2||f~_l - D_l a_l||^2 + gamma_l||a_l||_1] with the
        # code DETACHED (approved 2026-10-07, review N2): the gradient is the
        # partial derivative at the code, exactly what PC-D's weight update uses,
        # not a derivative through the 17 unrolled ISTA steps.
        return sum(wl * jnp.mean(lasso_energy(*stacked(f, Dl, xc, Gl, cfg)[:2], jax.lax.stop_gradient(a), gl,
                                              stacked(f, Dl, xc, Gl, cfg)[2]))
                   for f, (l, Dl, mu, Gl, xc), a, gl, wl in zip(ex["f_heads"], _heads(params, cfg, x), ex["a"],
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


def head_smooth(params, acts, a, cfg, x=None):
    """Smooth head energy (case C), batch mean: beta_g/(2B) sum ||z_s - D a||^2."""
    if is_D(cfg):
        sp = cfg.sparse
        e = 0.0
        for (l, Dl, mu, Gl, xc), al, wl in zip(_heads(params, cfg, x), a, sp.head_ws):
            t = 0.5 * sp.beta_g * jnp.sum((head_in(acts[l], mu) - al @ Dl.T) ** 2)
            if Gl is not None:           # case E anchor
                t = t + 0.5 * sp.beta_r * jnp.sum((xc - al @ Gl.T) ** 2)
            e = e + wl * t / acts[l].shape[0]
        return e
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


def a_step(params, acts, a, cfg, eta, x=None):
    """Case C proximal-gradient step on the code, at the PRE-step activities.

    a <- S_{eta_g gamma}(a - eta_g beta_g D^T (D a - z_s)),
    eta = eta_g from code_step(), computed once per relaxation.  The gradient
    is per sample (no B).
    """
    sp = cfg.sparse
    if is_D(cfg):           # one prox-gradient step per head; eta is a tuple
        out = []
        for (l, Dl, mu, Gl, xc), al, el, gl in zip(_heads(params, cfg, x), a, eta, sp.gammas):
            f = head_in(acts[l], mu)
            for _ in range(sp.code_k):   # code_k prox steps at the PRE-step activities
                gr = sp.beta_g * ((al @ Dl.T - f) @ Dl)
                if Gl is not None:       # case E anchor
                    gr = gr + sp.beta_r * ((al @ Gl.T - xc) @ Gl)
                al = soft_threshold(al - el * gr, el * gl)
            out.append(al)
        return tuple(out)
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


def health_D(params, acts, ex, cfg, n_fista=256):
    """Case D log scalars per head, and the collapse guards of the pilot gate.

    h{l}_fnorm        mean ||f_l|| of the RAW activity (scale guard; review D-M4/D1)
    h{l}_input_spread angular spread of f_l (collapse guard; review N3)
    h{l}_lasso        unweighted per-view head energy at the deployed code on f~_l
    h{l}_wlasso       w_l * h{l}_lasso  (x V gives the head's share of the BP-D objective)
    h{l}_subopt       relative LASSO suboptimality of a^(T_test) vs FISTA(500), first rows
    h{l}_D_spec       ||D_l||_2^2 (conditioning sets the ISTA step)
    h{l}_pr           participation ratio (sum lam)^2 / sum lam^2 of the covariance of the
                      CENTRED head layer f_l - mu_l: effective number of directions the
                      inputs vary along (rank-collapse guard)
    h{l}_mu_offset    ||mean_rows(f_l) - mu_l|| / rms ||f_l - mean|| (is mu tracking?)
    """
    sp = cfg.sparse
    z = acts[-1]
    out = {"embed_spread": _spread(z), "embed_norm": float(jnp.mean(jnp.linalg.norm(z, axis=-1))),
           "embed_std": float(jnp.mean(jnp.std(z, 0))), "n_rows": int(z.shape[0])}
    fr, sup = [], []
    Gs = get_G(params) if recon(cfg) else (None,) * len(sp.heads)
    xc = ex.get("x_c")
    for l, Dl, al, gl, wl, ft, mu, Gl in zip(sp.heads, get_D(params), ex["a"], sp.gammas, sp.head_ws,
                                             ex["f_heads"], get_mu(params), Gs):
        F, Dst, mst = stacked(ft, Dl, xc, Gl, cfg)        # == (ft, Dl, beta_g) for case D
        fc = np.asarray(acts[l]) - np.asarray(acts[l]).mean(0)
        lam = np.clip(np.linalg.eigvalsh(fc.T @ fc / fc.shape[0]), 0, None)
        pr = float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-30))
        off = float(np.linalg.norm(np.asarray(acts[l]).mean(0) - np.asarray(mu))
                    / (np.sqrt(np.mean(np.sum(fc ** 2, 1))) + 1e-12))
        on = np.asarray(al != 0)
        G = jnp.abs(Dl.T @ Dl) - jnp.eye(Dl.shape[1])
        e = float(jnp.mean(lasso_energy(F, Dst, al, gl, mst)))
        k = min(n_fista, ft.shape[0])
        e_star = float(jnp.mean(lasso_energy(F[:k], Dst, fista(F[:k], Dst, gl, mst, 500), gl, mst)))
        e_k = float(jnp.mean(lasso_energy(F[:k], Dst, al[:k], gl, mst)))
        h = {f"h{l}_active": float(on.mean()), f"h{l}_supports": len({r.tobytes() for r in on}),
             f"h{l}_dead_atoms": float(np.mean(~on.any(0))), f"h{l}_coherence": float(jnp.max(G)),
             f"h{l}_lasso": e, f"h{l}_wlasso": float(wl) * e,
             f"h{l}_rel_recon": float(jnp.mean(jnp.linalg.norm(ft - al @ Dl.T, axis=-1)
                                               / (jnp.linalg.norm(ft, axis=-1) + 1e-12))),
             f"h{l}_fnorm": float(jnp.mean(jnp.linalg.norm(acts[l], axis=-1))),
             f"h{l}_input_spread": _spread(acts[l]),
             f"h{l}_subopt": (e_k - e_star) / (abs(e_star) + 1e-12),
             f"h{l}_D_spec": float(spectral_sq(Dl)), f"h{l}_pr": pr, f"h{l}_mu_offset": off}
        # solver health (review N1): stacked Lipschitz, step, and the CODE-space distance
        # of the deployed a^(T) from the LASSO optimum (FISTA 2000) on the first k rows
        Lst = float(spectral_sq(Dst))
        a_opt = fista(F[:k], Dst, gl, mst, 2000)
        h[f"h{l}_stacked_L"] = Lst
        h[f"h{l}_eta"] = float(sp.step_frac / (mst * Lst))
        h[f"h{l}_code_err"] = float(jnp.linalg.norm(al[:k] - a_opt) / (jnp.linalg.norm(a_opt) + 1e-12))
        h[f"h{l}_active_opt"] = float(jnp.mean(a_opt != 0))
        if Gl is not None:     # case E: pixel R^2 of G_l a_l vs the anchor target (0 = mean image)
            res = np.asarray(xc - al @ Gl.T)
            h[f"h{l}_recon_r2"] = float(1.0 - np.sum(res ** 2) / (np.sum(np.asarray(xc) ** 2) + 1e-12))
            h[f"h{l}_G_spec"] = float(spectral_sq(Gl))
            # ENCODER-dependent reconstruction (review N3/D3): code from f~ ALONE (anchor off),
            # same D, gamma and step count; pixel info here must have come through the encoder
            eta_f = sp.step_frac / (sp.beta_g * spectral_sq(Dl))
            a_f = ista(ft, Dl, gl, sp.beta_g, t_deploy(cfg), eta_f)
            res_f = np.asarray(xc - a_f @ Gl.T)
            h[f"h{l}_recon_r2_enc"] = float(1.0 - np.sum(res_f ** 2) / (np.sum(np.asarray(xc) ** 2) + 1e-12))
            h[f"h{l}_recon_energy"] = float(0.5 * sp.beta_r * np.mean(np.sum(res ** 2, -1)))
            h[f"h{l}_head_energy"] = float(0.5 * sp.beta_g * jnp.mean(jnp.sum((ft - al @ Dl.T) ** 2, -1)))
        out.update(h)
        fr.append(h[f"h{l}_active"]); sup.append(h[f"h{l}_supports"])
    out["active_frac"] = float(np.mean(fr))
    out["active_min"], out["active_max"] = float(min(fr)), float(max(fr))
    out["supports_min"] = int(min(sup))
    # summed weighted head energy as the BP-D objective counts it (x V views, clean
    # centre rows): compare with loss_ff for the realised head/task ratio (review D5)
    out["head_energy_obj"] = float(cfg.data.n_views * sum(out[f"h{l}_wlasso"] for l in sp.heads))
    return out
