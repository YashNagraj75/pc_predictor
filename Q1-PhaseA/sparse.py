# Sparse-coding augmentations for the PC encoder: case B (soft-threshold a layer), case C (ISTA-coded
# layer with a trained dictionary), case D (per-head LASSO dictionaries explaining centred residual-stream
# activity, optionally anchored to pixels in case E). Provides dictionary/code init, ISTA/FISTA solvers,
# the deployed (test-time) forward pass, PC-relaxation hooks (dual rates, prox steps), and health/diagnostic
# scalars for the training log.
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
    """Returns whether sparse coding is active for cfg (case != "none")."""
    return cfg.sparse.case != "none"


def is_D(cfg) -> bool:
    """Returns whether cfg is case D: side dictionary heads on the residual stream, all multipliers kept."""
    return cfg.sparse.case == "D"


def head_in(v, mu):
    """Returns the centred, sqrt(n)-scaled unit vector (v - mu) / ||v - mu|| (safe norm), mu stop-gradiented.
    Centring (not raw v) avoids the encoder's shrink-to-offset collapse route (job 4faac68e)."""
    return normalise_rows(v - jax.lax.stop_gradient(mu), float(np.sqrt(v.shape[-1])))


def get_mu(params):
    """Returns the per-head running-mean mu array from params."""
    return params[3]


def recon(cfg) -> bool:
    """Returns whether cfg is case E: case D plus a pixel-reconstruction anchor (beta_r > 0).
    Params are then (model, skip, Ds, mus, Gs, xbar); beta_r=0 reduces exactly to case D's 4-tuple."""
    return is_D(cfg) and cfg.sparse.beta_r > 0


def get_G(params):
    """Returns the per-head generative dictionaries G from params (case E)."""
    return params[4]


def get_xbar(params):
    """Returns the fixed per-pixel training mean xbar from params (case E)."""
    return params[5]


def stacked(f_t, Dl, xc, Gl, cfg):
    """Returns the per-head LASSO (target, dictionary, mu): plain (f~, D, beta_g) for case D, or the
    beta-weighted stack [sqrt(bg) f~; sqrt(br) xc], [sqrt(bg) D; sqrt(br) G] for case E's anchor."""
    sp = cfg.sparse
    if not recon(cfg):
        return f_t, Dl, sp.beta_g
    return (jnp.concatenate([jnp.sqrt(sp.beta_g) * f_t, jnp.sqrt(sp.beta_r) * xc], -1),
            stacked_D(Dl, Gl, cfg), 1.0)


def stacked_D(Dl, Gl, cfg):
    """Returns the case-E stacked dictionary [sqrt(beta_g) D; sqrt(beta_r) G]."""
    return jnp.concatenate([jnp.sqrt(cfg.sparse.beta_g) * Dl, jnp.sqrt(cfg.sparse.beta_r) * Gl], 0)


def _heads(params, cfg, x=None):
    """Returns, per head, the tuple (layer, D_l, mu_l, G_l or None, centred pixels or None)."""
    sp = cfg.sparse
    Ds, mus = get_D(params), get_mu(params)
    if recon(cfg):
        assert x is not None, "case E needs the input view"
        xc = anchor_target(x, get_xbar(params))
        return [(l, Dl, mu, Gl, xc) for l, Dl, mu, Gl in zip(sp.heads, Ds, mus, get_G(params))]
    return [(l, Dl, mu, None, None) for l, Dl, mu in zip(sp.heads, Ds, mus)]


def mu_rows(Xtr, cfg, n=1024):
    """Returns calibration rows (rng(0), same images across calls) for init_mu and gamma/beta_r selection:
    training centre views for case D, one fixed-key augmented view per image for case E."""
    import data as Dd
    idx = np.random.default_rng(0).choice(len(Xtr), n, replace=False)
    if recon(cfg):
        views, _ = Dd.make_views(jax.random.PRNGKey(0), jnp.asarray(Xtr[idx]), cfg.data)
        return views[0]
    return Dd.eval_batch(jnp.asarray(Xtr[idx]), cfg.data)


def t_deploy(cfg):
    """Returns the number of deployed ISTA steps after a^(0): T_test * code_k."""
    return cfg.sparse.T_test * cfg.sparse.code_k


def anchor_target(x, xbar):
    """Returns the case-E reconstruction target: x minus the fixed pixel mean xbar, then minus its own per-view mean."""
    xc = x - jax.lax.stop_gradient(xbar)
    return xc - jnp.mean(xc, axis=-1, keepdims=True)


def _head_means(params, x, cfg):
    """Returns the feedforward per-head activation means over x."""
    z, hs = E.forward(params, x, cfg.enc)
    acts = [*hs, z]
    return tuple(jnp.mean(acts[l], axis=0) for l in cfg.sparse.heads)


def init_mu(params, x, cfg):
    """Sets mu_l to the feedforward mean of head layer l over x (and xbar for case E).
    Call before training, before head selection, and for every untrained floor."""
    if not is_D(cfg):
        return params
    out = list(params)
    out[3] = _head_means(params, x, cfg)
    if recon(cfg):
        out[5] = jnp.mean(x, axis=0)          # per-pixel training mean, fixed after this
    return tuple(out)


@eqx.filter_jit
def _mu_ema(params, Zin, cfg):
    """Returns mu updated by one EMA step toward this batch's feedforward per-head means."""
    m = cfg.sparse.mu_momentum
    means = [jnp.mean(jnp.stack([_head_means(params, Zin[v], cfg)[h] for v in range(Zin.shape[0])]), axis=0)
             for h in range(len(cfg.sparse.heads))]
    return tuple(m * mu + (1 - m) * jax.lax.stop_gradient(nm) for mu, nm in zip(params[3], means))


def update_mu(params, Zin, cfg):
    """Returns params with mu EMA-updated from this step's training views, after the optimiser step."""
    if not is_D(cfg):
        return params
    out = list(params)
    out[3] = _mu_ema(params, Zin, cfg)
    return tuple(out)


def check_D(cfg, L):
    """Asserts that case D's heads/head_ws/gammas are consistent and interior to the network."""
    sp = cfg.sparse
    assert len(sp.heads) >= 1, "case D needs at least one head"
    assert len(sp.head_ws) == len(sp.heads) == len(sp.gammas), \
        f"heads/head_ws/gammas lengths differ: {sp.heads} {sp.head_ws} {sp.gammas}"
    assert len(set(sp.heads)) == len(sp.heads), f"duplicate heads {sp.heads}"
    assert all(0 < l < L - 1 for l in sp.heads), f"heads {sp.heads} must be interior (1..{L-2})"


def soft_threshold(v, theta):
    """Returns the soft-threshold S_theta(v) = sign(v) * max(|v| - theta, 0)."""
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - theta, 0.0)


def normalise_dict(D, eps=1e-8):
    """Normalizes D's columns to unit norm."""
    return D / jnp.maximum(jnp.linalg.norm(D, axis=0, keepdims=True), eps)


def spectral_sq(D):
    """Returns ||D||_2^2 (exact spectral norm squared) for the solver step size; D is stop-gradiented.
    Stopping only the output still built a JVP, so stop_gradient is applied to D itself."""
    Ds = jax.lax.stop_gradient(D)
    # exact: lambda_max(D D^T) = sigma_max(D)^2, via a symmetric eigensolve of the
    # n x n Gram, cheaper on GPU than the SVD behind norm(ord=2) (review D-M5/N7)
    # n x n Gram (or m x m when D is tall, e.g. case E's stacked [D; G]: same value)
    return jnp.linalg.eigvalsh(Ds @ Ds.T if Ds.shape[0] <= Ds.shape[1] else Ds.T @ Ds)[-1]


def code_step(params, cfg):
    """Returns the ISTA step size eta_g = step_frac / (beta_g ||D||^2), per head for case D.
    Computed once per relaxation; head weights w_l cancel from the code dynamics."""
    if recon(cfg):          # exact Lipschitz constant of the stacked LASSO
        sp = cfg.sparse
        return tuple(sp.step_frac / spectral_sq(stacked_D(Dl, Gl, cfg))
                     for Dl, Gl in zip(get_D(params), get_G(params)))
    if is_D(cfg):
        return tuple(cfg.sparse.step_frac / (cfg.sparse.beta_g * spectral_sq(Dl))
                     for Dl in get_D(params))
    return cfg.sparse.step_frac / (cfg.sparse.beta_g * spectral_sq(get_D(params)))


def huber_energy(f, beta, rho):
    """Returns the per-sample Moreau envelope of beta||.||_1 at f, shape (B,)."""
    zs = soft_threshold(f, beta / rho)
    return 0.5 * rho * jnp.sum((f - zs) ** 2, -1) + beta * jnp.sum(jnp.abs(zs), -1)


def lasso_mu(cfg):
    """Returns the LASSO coupling mu = rho * beta_g / (rho + beta_g)."""
    rho, bg = cfg.pc.rho, cfg.sparse.beta_g
    return rho * bg / (rho + bg)


# ------------------------------------------------------------------ params

def init_params(key, cfg, input_dim):
    """Returns initial params: E.init's encoder plus the case-specific architecture change and dictionaries.
    Shared layers use the same key as E.init; dictionaries come from independent folds of that key."""
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
    """Re-normalises D's (and case E's G's) columns to unit norm after an optimiser step."""
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
    """Returns the dictionary D (or per-head dictionaries) from params."""
    return params[2]


def param_labels(params):
    """Returns optax.multi_transform labels ("enc"/"dict"/"frozen") for each param leaf."""
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
    """Runs t ISTA steps minimizing mu/2||f - D a||^2 + gamma||a||_1 (rows of f independent).
    a0 defaults to one ISTA step from a=0: S_(eta*gamma)(eta*mu*D^T f)."""
    th = eta * gamma
    if a0 is None:
        a0 = soft_threshold(eta * mu * (f @ D), th)

    def step(_, a):
        """Returns one ISTA update of code a."""
        return soft_threshold(a + eta * mu * ((f - a @ D.T) @ D), th)

    return jax.lax.fori_loop(0, t, step, a0) if t > 0 else a0


def fista(f, D, gamma, mu, n_iter=2000):
    """Runs n_iter FISTA steps as a reference LASSO solver, with step size exactly 1/(mu * ||D||^2)."""
    eta = 1.0 / (mu * spectral_sq(D))
    th = eta * gamma

    def step(_, c):
        """Returns one FISTA (accelerated proximal-gradient) update of (a, y, momentum tk)."""
        a, y, tk = c
        a_new = soft_threshold(y + eta * mu * ((f - y @ D.T) @ D), th)
        tk1 = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * tk ** 2))
        return a_new, a_new + ((tk - 1.0) / tk1) * (a_new - a), tk1

    a0 = jnp.zeros((f.shape[0], D.shape[1]), f.dtype)
    a, _, _ = jax.lax.fori_loop(0, n_iter, step, (a0, a0, jnp.asarray(1.0, f.dtype)))
    return a


def lasso_energy(f, D, a, gamma, mu):
    """Returns the per-sample LASSO loss mu/2 * ||f - D a||^2 + gamma * ||a||_1, shape (B,)."""
    return 0.5 * mu * jnp.sum((f - a @ D.T) ** 2, -1) + gamma * jnp.sum(jnp.abs(a), -1)


def kkt_violation(f, D, a, gamma, mu):
    """Returns the per-sample max violation of the LASSO KKT optimality conditions, divided by gamma."""
    c = mu * ((f - a @ D.T) @ D)
    on = jnp.abs(c - gamma * jnp.sign(a))
    off = jnp.maximum(jnp.abs(c) - gamma, 0.0)
    v = jnp.where(a != 0, on, off)
    return jnp.max(v, -1) / jnp.maximum(gamma, 1e-12)


# ------------------------------------------------------- deployed forward

def _scalings(params, x, cfg):
    """Returns jpc's per-layer parameter scalings for the deployed network at x."""
    model, skip = params[0], params[1]
    return jpc._get_param_scalings(model=model, input=x, skip_model=skip,
                                   param_type=cfg.enc.parameterisation,
                                   gamma=E.gamma_for(cfg.enc))


def norm_spec(cfg):
    """Returns (layer, norm constant) if the sparse layer's prediction is normalised, else None."""
    sp = cfg.sparse
    if sp.case in ("none", "D") or not sp.normalise_fs:
        return None
    return (sp.layer, sp.fs_norm)


NORM_EPS = 1e-6


def normalise_rows(f, c):
    """Returns c * f / sqrt(||f||^2 + eps^2) per row (a safe norm, avoiding 0/0 on all-zero rows, job b3c717c6)."""
    return c * f / jnp.sqrt(jnp.sum(f * f, axis=-1, keepdims=True) + NORM_EPS ** 2)


def _pred(params, scalings, l, prev, norm=None):
    """Returns f_l(prev), mirroring jpc's term structure; normalises layer s's prediction if norm=(s, c) matches l."""
    model, skip = params[0], params[1]
    p = scalings[l] * vmap(model[l])(prev)
    if skip[l] is not None:
        p = p + vmap(skip[l])(prev)
    if norm is not None and l == norm[0]:
        p = normalise_rows(p, norm[1])
    return p


def deploy(params, x, cfg, t=None, beta_g=None):
    """Runs the deployed (test-time) network, returning (acts, extras): each head codes its own LASSO for
    case D, case B/C threshold/code the sparse layer; t overrides ISTA steps, beta_g=0 forces a plain pass."""
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
    """Returns (z, hs) in encoder.forward's contract, from the deployed network."""
    acts, _ = deploy(params, x, cfg, t=t)
    return acts[-1], acts[:-1]


def deployed_head_energy(params, x, cfg):
    """Returns the deployed sparse-layer energy for one view, batch mean (Huber for B, LASSO for C,
    weighted per-head LASSO for D); train.bp_objective sums this over views."""
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
    """Returns per-layer dual rates: zero below/at the sparse layer (unless below_hard) for B/C, alpha
    everywhere for case D/none."""
    a = cfg.pc.alpha
    if not active(cfg) or is_D(cfg):          # case D keeps every multiplier
        return [a] * L
    s = cfg.sparse.layer
    return [0.0 if (l == s or (not cfg.sparse.below_hard and l < s)) else a
            for l in range(L)]


def relax_init(params, x, cfg):
    """Returns the relaxation start for one view: the deployed forward pass at t=0, as (acts, a or None)."""
    acts, ex = deploy(params, x, cfg, t=0)
    return acts, ex.get("a")


def head_smooth(params, acts, a, cfg, x=None):
    """Returns the smooth case-C/D head energy, batch mean: beta_g/2 ||z_s - D a||^2 (plus case E's anchor term)."""
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
    """Returns the L1 sparsity term, batch mean (reported only, never differentiated)."""
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
    """Returns acts with case B's proximal soft-threshold step applied to layer s, using layer s's own step size."""
    if cfg.sparse.case != "B":
        return acts
    s = cfg.sparse.layer
    out = list(acts)
    out[s] = soft_threshold(acts[s], cfg.sparse.s_step / cfg.pc.rho * cfg.sparse.beta)
    return out


def a_step(params, acts, a, cfg, eta, x=None):
    """Returns one (or code_k, for case D) proximal-gradient step(s) on code a, at pre-step activities.
    eta is code_step()'s per-relaxation step size; the gradient is per-sample, not batch-averaged."""
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
    """Returns the fraction of nonzero units in the sparse variable (z_s for B, a for C; mean over heads for D)."""
    sp = cfg.sparse
    if is_D(cfg):           # mean over heads
        return sum(jnp.mean((al != 0).astype(jnp.float32)) for al in a) / len(a)
    v = acts[sp.layer] if sp.case == "B" else a
    return jnp.mean((v != 0).astype(jnp.float32))


# ------------------------------------------------------------- health

def health(params, x, cfg):
    """Returns training-log health scalars (rescaling, sparsity, dictionary diagnostics) for the deployed network."""
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
    """Returns 1 minus the median cosine of each row to the mean direction (0 = all rows point the same way)."""
    m = jnp.mean(v, 0)
    c = (v @ m) / (jnp.linalg.norm(v, axis=1) * jnp.linalg.norm(m) + 1e-12)
    return float(1.0 - jnp.median(c))


def health_D(params, acts, ex, cfg, n_fista=256):
    """Returns case-D per-head log scalars and collapse guards: activity scale/spread, LASSO energy and
    suboptimality vs FISTA(500), dictionary coherence/spectral norm, participation ratio, mu offset, etc."""
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
