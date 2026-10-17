import dataclasses, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")    # review N1: no TF32 on the Ampere GPU
import jax.numpy as jnp, equinox as eqx, optax, jpc
import pred_ff as PF

M, DT = 128, 10
WIDTH = {3: 600, 10: 256}
C_GRID = (0.25, 0.5, 1.0, 2.0)
RHO_GRID = (1.0, 4.0, 16.0)      # alm penalty (= hidden precision); C2 selection grid
C_ALM = (0.25, 0.5, 1.0)       # c = 2 diverges for rho >= 4 (gate, 2026-10-17)


@dataclasses.dataclass
class GCfg:
    backbone: str = "bp"
    rule: str = "pc"            # pc | bp | alm (PC-ALM: multipliers on the hidden levels)
    start: str = "zero"         # zero | amort
    depth: int = 3
    width: int = 0              # 0 -> WIDTH[depth]
    hid_prec: float = 1.0       # precision of the hidden levels (sensitivity arm: 9 at depth 10)
    seed: int = 0
    c: float = 1.0
    T: int = 400                # inference steps on the seen views (training and default prediction)
    T2: int = 100               # PC phase 2: hidden-level relaxation of all views at fixed s
    prior: float = 1.0
    out_w: float = 1.0
    steps: int = 4000
    stop_at: int = 0            # >0: stop early but keep the full-length schedule (step-scale selection)
    batch: int = 256
    lr: float = 1e-3
    wd: float = 1e-4
    T_amort: int = 20
    alpha: float = 0.0          # alm: dual rate, lam_l += alpha * r_l
    k_inner: int = 1            # alm: primal steps per dual step (total primal steps stays T)
    log_align: bool = True      # credit diagnostic at log steps (pc / alm)

    @property
    def H(self):
        return self.depth - 1

    @property
    def n(self):
        return self.width or WIDTH.get(self.depth, 256)

    @property
    def name(self):
        tag = (f"_w{self.width}" if self.width else "") + (f"_hp{self.hid_prec:g}" if self.hid_prec != 1.0 else "")
        tag += f"_a{self.alpha:g}_k{self.k_inner}" if self.rule == "alm" else ""
        return f"gpc_{self.rule}_d{self.depth}_{self.start}{tag}"


def phi(tau):
    return jnp.concatenate([tau, jnp.ones(tau.shape[:-1] + (1,), tau.dtype)], -1)


# ------------------------------------------------------------------ parameters
def init(key, cfg, d=128):
    n, H = cfg.n, cfg.H
    ks = jax.random.split(key, H + 1)
    g = lambda k, o, i: jax.random.normal(k, (o, i)) / math.sqrt(i)
    return {"top": (g(ks[0], n, M + DT), jnp.zeros(n)),
            "mid": [g(ks[1 + j], n, n) for j in range(H - 1)],          # mid[j] predicts x_{j+1} from x_{j+2}
            "bot": (g(ks[H], d, n), jnp.zeros(d))}


def init_amort(key, d=128):
    return jpc.make_mlp(key, d + 9, 256, 2, M, "relu", use_bias=True, param_type="sp")


def n_params(p):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(p)))


# ------------------------------------------------------------------ predictions and energy
def mu_top(p, s, Ph):
    k = Ph.shape[1]
    inp = jnp.concatenate([jnp.broadcast_to(s[:, None], (s.shape[0], k, s.shape[-1])), Ph], -1)
    return inp @ p["top"][0].T + p["top"][1]


def mu_mid(W, x, H):
    return x + jax.nn.relu(x) @ W.T / math.sqrt(H)


def mu_bot(p, x1):
    return jax.nn.relu(x1) @ p["bot"][0].T + p["bot"][1]


def topdown(p, s, Ph, H):
    xs = [None] * H
    xs[H - 1] = mu_top(p, s, Ph)
    for l in range(H - 1, 0, -1):
        xs[l - 1] = mu_mid(p["mid"][l - 1], xs[l], H)
    return xs, mu_bot(p, xs[0])


# Unweighted per-level energies [level 0 (observation), 1, ..., H]
# summed over views and batch.
def level_energies(p, s, xs, Z, Ph, H):
    e = [0.5 * jnp.sum((Z - mu_bot(p, xs[0])) ** 2)]
    for l in range(1, H):
        e.append(0.5 * jnp.sum((xs[l - 1] - mu_mid(p["mid"][l - 1], xs[l], H)) ** 2))
    e.append(0.5 * jnp.sum((xs[H - 1] - mu_top(p, s, Ph)) ** 2))
    return e


def energy(p, s, xs, Z, Ph, H, prior, out_w=1.0, hid=1.0):
    e = level_energies(p, s, xs, Z, Ph, H)
    return out_w * e[0] + hid * sum(e[1:]) + 0.5 * prior * jnp.sum(s ** 2)


# Hidden-level residuals [r_1 .. r_H], same order as level_energies[1:]; r_H = x_H - mu_top(s).
def hid_residuals(p, s, xs, Ph, H):
    r = [xs[l - 1] - mu_mid(p["mid"][l - 1], xs[l], H) for l in range(1, H)]
    return r + [xs[H - 1] - mu_top(p, s, Ph)]


# PC-ALM augmented energy (completing-the-square form, rho = hid): the observation term is the objective,
# every hidden residual carries a multiplier.  lam = None or all zero gives energy() exactly (gated).
def alm_energy(p, s, xs, Z, Ph, H, prior, out_w, hid, lam):
    r = hid_residuals(p, s, xs, Ph, H)
    if lam is not None:
        r = [ri + li for ri, li in zip(r, lam)]
    e0 = 0.5 * jnp.sum((Z - mu_bot(p, xs[0])) ** 2)
    return out_w * e0 + hid * 0.5 * sum(jnp.sum(ri ** 2) for ri in r) + 0.5 * prior * jnp.sum(s ** 2)


def ad_loss(p, s, Z, Ph, H, prior):
    return 0.5 * jnp.sum((Z - topdown(p, s, Ph, H)[1]) ** 2) + 0.5 * prior * jnp.sum(s ** 2)


def _E(cfg):
    return lambda p, s, xs, Z, Ph: energy(p, s, xs, Z, Ph, cfg.H, cfg.prior, cfg.out_w, cfg.hid_prec)


# ------------------------------------------------------------------ step sizes
# Spectral norm via power iteration from a fixed start (review N5/G9: no SVD in step)
# Applies 2% safety margin
def _sn(W, iters=20):
    W = jax.lax.stop_gradient(W)
    v = jax.random.normal(jax.random.PRNGKey(17), (W.shape[1],))
    v = v / jnp.linalg.norm(v)
    for _ in range(iters):
        v = W.T @ (W @ v)
        v = v / (jnp.linalg.norm(v) + 1e-30)
    return 1.02 * jnp.linalg.norm(W @ v)


# [eta_s, eta_x1..eta_xH] = c * 0.9 / (own curvature + 2 * curvature of the term below).
def pc_etas(p, k, cfg):
    H, h = cfg.H, cfg.hid_prec
    below = [cfg.out_w * _sn(p["bot"][0]) ** 2] + [h * (1.0 + _sn(W) / math.sqrt(H)) ** 2 for W in p["mid"]]
    eta_x = [cfg.c * 0.9 / (h + 2.0 * L) for L in below]
    eta_s = cfg.c * 0.9 / (cfg.prior + 2.0 * h * k * _sn(p["top"][0][:, :M]) ** 2)
    return [eta_s] + eta_x


# Per-sample lambda_max(J^T J), where J = dG/ds evaluated at s0.
# Computed via batched power iteration from a fixed random start.
def gn_lambda(p, s0, Ph, H, iters=25):
    p, s0 = jax.lax.stop_gradient(p), jax.lax.stop_gradient(s0)
    f = lambda s: topdown(p, s, Ph, H)[1]
    _, vjp = jax.vjp(f, s0)
    # SAME start vector for every sample (a per-row random start made each sample's step depend on its position
    # in the batch: reordering a batch changed BP inference by 11% on the GPU, 2026-10-13)
    v = jnp.broadcast_to(jnp.linspace(1.0, 2.0, s0.shape[-1]), s0.shape)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    def body(_, v):
        w = vjp(jax.jvp(f, (s0,), (v,))[1])[0]
        return w / (jnp.linalg.norm(w, axis=-1, keepdims=True) + 1e-12)
    v = jax.lax.fori_loop(0, iters, body, v)
    w = vjp(jax.jvp(f, (s0,), (v,))[1])[0]
    return jnp.linalg.norm(w, axis=-1)


def ad_eta(p, s0, Ph, cfg):
    return (cfg.c * 0.9 / (cfg.prior + 2.0 * gn_lambda(p, s0, Ph, cfg.H)))[:, None]


# ------------------------------------------------------------------ inference
def start_s(p, amort, Z, T9, cfg):
    b = Z.shape[0]
    if cfg.start == "zero" or amort is None:
        return jnp.zeros((b, M))
    k = Z.shape[1]
    x = jnp.concatenate([Z, T9], -1).reshape(b * k, -1)
    return PF.forward(amort, x).reshape(b, k, M).mean(1)


# Returns: s, xs, energy history (T,), size of next s step relative to |s| (convergence diagnostic).
def pc_infer(p, s0, Z, Ph, cfg, T=None, freeze_s=False, xs0=None):
    T = cfg.T if T is None else T
    xs0 = topdown(p, s0, Ph, cfg.H)[0] if xs0 is None else xs0
    etas = pc_etas(p, Z.shape[1], cfg)
    Ef = _E(cfg)
    gfn = jax.grad(lambda v: Ef(p, v[0], v[1], Z, Ph))

    def step(v, _):
        gs, gx = gfn(v)
        s = v[0] if freeze_s else v[0] - etas[0] * gs
        xs = [x - e * g for x, g, e in zip(v[1], gx, etas[1:])]
        return (s, xs), Ef(p, s, xs, Z, Ph)

    (s, xs), hist = jax.lax.scan(step, (s0, xs0), None, length=T)
    gs = gfn((s, xs))[0]
    rel = jnp.mean(jnp.linalg.norm(etas[0] * gs, axis=-1) / (jnp.linalg.norm(s, axis=-1) + 1e-8))
    return s, xs, hist, rel


# PC-ALM relaxation: T primal steps on (s, xs) with the PC step sizes, a dual step lam += alpha * r after every
# k_inner primal steps (never after the last).  alpha = 0 is pc_infer exactly (gated).  At the converged saddle
# r = 0, s minimises the auto-decoder loss and the weight gradient equals the BP auto-decoder's.
def alm_infer(p, s0, Z, Ph, cfg, T=None, freeze_s=False, xs0=None, lam0=None):
    T = cfg.T if T is None else T
    H = cfg.H
    xs0 = topdown(p, s0, Ph, H)[0] if xs0 is None else xs0
    lam0 = [jnp.zeros_like(x) for x in xs0] if lam0 is None else lam0
    etas = pc_etas(p, Z.shape[1], cfg)
    Lf = lambda s, xs, lam: alm_energy(p, s, xs, Z, Ph, H, cfg.prior, cfg.out_w, cfg.hid_prec, lam)
    gfn = jax.grad(lambda v, lam: Lf(v[0], v[1], lam))

    def step(c, t):
        (s, xs), lam = c
        gs, gx = gfn((s, xs), lam)
        s = s if freeze_s else s - etas[0] * gs
        xs = [x - e * g for x, g, e in zip(xs, gx, etas[1:])]
        do = ((t + 1) % cfg.k_inner == 0) & (t < T - 1)
        r = hid_residuals(p, s, xs, Ph, H)
        lam = [jnp.where(do, l + cfg.alpha * ri, l) for l, ri in zip(lam, r)]
        return ((s, xs), lam), Lf(s, xs, lam)

    ((s, xs), lam), hist = jax.lax.scan(step, ((s0, xs0), lam0), jnp.arange(T))
    gs = gfn((s, xs), lam)[0]
    rel = jnp.mean(jnp.linalg.norm(etas[0] * gs, axis=-1) / (jnp.linalg.norm(s, axis=-1) + 1e-8))
    return s, xs, lam, hist, rel


def alm_wgrads(p, s, xs, lam, Z, Ph, cfg):
    s, xs, lam = jax.lax.stop_gradient(s), jax.lax.stop_gradient(xs), jax.lax.stop_gradient(lam)
    g = jax.grad(lambda q: alm_energy(q, s, xs, Z, Ph, cfg.H, cfg.prior, cfg.out_w, cfg.hid_prec, lam))(p)
    return jax.tree_util.tree_map(lambda x: x / Z.shape[0], g)


def ad_infer(p, s0, Z, Ph, cfg, T=None):
    T = cfg.T if T is None else T
    eta = ad_eta(p, s0, Ph, cfg)
    gfn = jax.grad(lambda s: ad_loss(p, s, Z, Ph, cfg.H, cfg.prior))

    def step(s, _):
        s = s - eta * gfn(s)
        return s, ad_loss(p, s, Z, Ph, cfg.H, cfg.prior)

    s, hist = jax.lax.scan(step, s0, None, length=T)
    rel = jnp.mean(jnp.linalg.norm(eta * gfn(s), axis=-1) / (jnp.linalg.norm(s, axis=-1) + 1e-8))
    return s, hist, rel


# Belief from the given (seen) views: returns (s*, s0, energy history, relative last step).
def infer_s(p, amort, Z, T9, cfg, T=None):
    Ph = phi(T9)
    s0 = start_s(p, amort, Z, T9, cfg)
    if T == 0:
        return s0, s0, jnp.zeros(1), jnp.zeros(())
    if cfg.rule == "pc":
        s, _, hist, rel = pc_infer(p, s0, Z, Ph, cfg, T)
    elif cfg.rule == "alm":
        s, _, _, hist, rel = alm_infer(p, s0, Z, Ph, cfg, T)
    else:
        s, hist, rel = ad_infer(p, s0, Z, Ph, cfg, T)
    return s, s0, hist, rel


# ------------------------------------------------------------------ weight gradients
def pc_wgrads(p, s, xs, Z, Ph, cfg):
    s, xs = jax.lax.stop_gradient(s), jax.lax.stop_gradient(xs)
    g = jax.grad(lambda q: _E(cfg)(q, s, xs, Z, Ph))(p)
    return jax.tree_util.tree_map(lambda x: x / Z.shape[0], g)


def ad_wgrads(p, s, Z, Ph, cfg):
    s = jax.lax.stop_gradient(s)
    g = jax.grad(lambda q: ad_loss(q, s, Z, Ph, cfg.H, cfg.prior))(p)
    return jax.tree_util.tree_map(lambda x: x / Z.shape[0], g)


def amort_grads(amort, Z, T9, s, cfg):
    b, k = Z.shape[:2]
    x = jnp.concatenate([Z, T9], -1).reshape(b * k, -1)
    y = jnp.repeat(jax.lax.stop_gradient(s)[:, None], k, 1).reshape(b * k, M)
    if cfg.rule == "pc":
        return PF.pc_grads(amort, x, y, PF.Cfg(T=cfg.T_amort, adaptive_eta=True))
    if cfg.rule == "alm":
        return PF.alm_grads(amort, x, y, PF.Cfg(T=cfg.T_amort, adaptive_eta=True, alpha=cfg.alpha))
    return PF.bp_grads(amort, x, y, None)


# PC phase 2 / BP: gradients on all 4 views at the belief s inferred from the seen views.
# Phase 2: all views relaxed at fixed s, seen views warm-started from phase 1 (pc: xs; alm: (xs, lam)).
# Returns the phase-2 state and its energy history.
def phase2(p, s, carry, Zall, Pall, seen, cfg):
    ii = jnp.arange(Zall.shape[0])[:, None]
    xs_s, lam_s = carry if cfg.rule == "alm" else (carry, None)
    xs0 = topdown(p, s, Pall, cfg.H)[0]
    xs0 = [x.at[ii, seen].set(xv) for x, xv in zip(xs0, xs_s)]
    if cfg.rule == "alm":
        lam0 = [jnp.zeros_like(x).at[ii, seen].set(lv) for x, lv in zip(xs0, lam_s)]
        _, xs, lam, h2, _ = alm_infer(p, s, Zall, Pall, cfg, T=cfg.T2, freeze_s=True, xs0=xs0, lam0=lam0)
        return xs, lam, h2
    _, xs, h2, _ = pc_infer(p, s, Zall, Pall, cfg, T=cfg.T2, freeze_s=True, xs0=xs0)
    return xs, None, h2


def weight_grads(p, s, Zs_seen_xs, Zall, Pall, seen, cfg):
    if cfg.rule == "pc":
        xs, _, h2 = phase2(p, s, Zs_seen_xs, Zall, Pall, seen, cfg)
        return pc_wgrads(p, s, xs, Zall, Pall, cfg), h2
    if cfg.rule == "alm":
        xs, lam, _ = phase2(p, s, Zs_seen_xs, Zall, Pall, seen, cfg)
        return alm_wgrads(p, s, xs, lam, Zall, Pall, cfg), _E(cfg)(p, s, xs, Zall, Pall)[None]   # plain energy
    return ad_wgrads(p, s, Zall, Pall, cfg), jnp.zeros(1)


# ------------------------------------------------------------------ training
def train(cfg, ztr, ttr, verbose=True, log_every=250, ckpt=None):
    assert cfg.rule != "alm" or cfg.out_w == 1.0, "alm saddle equals the BP auto-decoder only at out_w = 1"
    kp, ka = jax.random.split(jax.random.PRNGKey(cfg.seed))
    p = init(kp, cfg, ztr.shape[-1])
    amort = init_amort(ka, ztr.shape[-1]) if cfg.start == "amort" else None
    mk = lambda: optax.chain(optax.clip_by_global_norm(1.0),
                             optax.adamw(optax.warmup_cosine_decay_schedule(0.0, cfg.lr, min(200, cfg.steps // 4), cfg.steps),
                                         weight_decay=cfg.wd))
    opt, opt_a = mk(), mk()
    st = opt.init(p)
    st_a = opt_a.init(eqx.filter(amort, eqx.is_inexact_array)) if amort is not None else None

    @eqx.filter_jit
    def step(p, amort, st, st_a, Zall, Tall, perm, k):
        ii = jnp.arange(Zall.shape[0])[:, None]
        seen, tgt = perm[:, :k], perm[:, k:]
        Zs, Ts = Zall[ii, seen], Tall[ii, seen]
        Ps, Pall = phi(Ts), phi(Tall)
        s0 = start_s(p, amort, Zs, Ts, cfg)
        E_init = _E(cfg)(p, s0, topdown(p, s0, Ps, cfg.H)[0], Zs, Ps) if cfg.rule in ("pc", "alm") else ad_loss(p, s0, Zs, Ps, cfg.H, cfg.prior)
        if cfg.rule == "pc":
            s, xs_seen, hist, rel = pc_infer(p, s0, Zs, Ps, cfg)
        elif cfg.rule == "alm":
            s, xs_a, lam_a, hist, rel = alm_infer(p, s0, Zs, Ps, cfg)
            xs_seen = (xs_a, lam_a)
        else:
            s, hist, rel = ad_infer(p, s0, Zs, Ps, cfg)
            xs_seen = None
        g, h2 = weight_grads(p, s, xs_seen, Zall, Pall, seen, cfg)
        gn = optax.global_norm(g)
        Zhat = topdown(p, jax.lax.stop_gradient(s), phi(Tall[ii, tgt]), cfg.H)[1]
        pred_err = 0.5 * jnp.sum((Zall[ii, tgt] - Zhat) ** 2) / Zall.shape[0]   # held-out views of this batch
        upd, st = opt.update(g, st, p)
        p2 = optax.apply_updates(p, upd)
        a_err, amort2, st_a2 = jnp.nan, amort, st_a
        if amort is not None:
            ga = amort_grads(amort, Zs, Ts, s, cfg)
            upd_a, st_a2 = opt_a.update(ga, st_a, eqx.filter(amort, eqx.is_inexact_array))
            amort2 = eqx.apply_updates(amort, upd_a)
            a_err = jnp.mean(jnp.sum((s0 - s) ** 2, -1)) / (jnp.mean(jnp.sum(s ** 2, -1)) + 1e-12)
        finite = jnp.isfinite(hist[-1]) & jnp.isfinite(gn) & jnp.isfinite(pred_err)
        up = jnp.mean((jnp.diff(hist) > 1e-3 * jnp.abs(hist[0])).astype(jnp.float32)) if cfg.rule != "alm" else jnp.nan
        if cfg.rule == "alm":
            nx = jnp.sqrt(sum(jnp.sum(x ** 2) for x in xs_seen[0])) + 1e-30
            res_rel = jnp.sqrt(sum(jnp.sum(x ** 2) for x in hid_residuals(p, s, xs_seen[0], Ps, cfg.H))) / nx
            lam_rel = jnp.sqrt(sum(jnp.sum(x ** 2) for x in xs_seen[1])) / nx
        else:
            res_rel = lam_rel = jnp.nan
        # alm: hist is the AUGMENTED energy (rises as the duals grow), so log the plain energy at the final state
        E_fin = _E(cfg)(p, s, xs_seen[0], Zs, Ps) if cfg.rule == "alm" else hist[-1]
        m = {"E_init": E_init / Zs.shape[0], "E": E_fin / Zs.shape[0], "pred_err": pred_err, "gnorm": gn,
             "rel_last_step_s": rel, "frac_up_steps": up, "s_norm": jnp.mean(jnp.linalg.norm(s, axis=-1)),
             "amort_rel_err": a_err, "E2": h2[-1] / Zall.shape[0], "res_rel": res_rel, "lam_rel": lam_rel, "finite": finite}
        return p2, amort2, st, st_a2, m

    # Credit diagnostic (pc / alm only; never feeds training), on a FIXED probe batch with fixed seen views, k = 1 and 3:
    # cosine of this rule's weight gradient with the BP auto-decoder gradient at the SAME s (all / seen / unseen views,
    # per block), norm ratio, phase-2 residual size, and distance of s from a settled auto-decoder belief (c=1, T=2000).
    @eqx.filter_jit
    def align(p, amort, Zall, Tall, perm, k):
        ii = jnp.arange(Zall.shape[0])[:, None]
        seen, uns = perm[:, :k], perm[:, k:]
        Zs, Ts = Zall[ii, seen], Tall[ii, seen]
        Ps, Pall = phi(Ts), phi(Tall)
        s0 = start_s(p, amort, Zs, Ts, cfg)
        if cfg.rule == "alm":
            s, xs, lam, _, _ = alm_infer(p, s0, Zs, Ps, cfg); carry = (xs, lam)
        else:
            s, xs, _, _ = pc_infer(p, s0, Zs, Ps, cfg); carry = xs
        xf, lf, _ = phase2(p, s, carry, Zall, Pall, seen, cfg)
        def grads(ix):
            sel = (lambda a: a) if ix is None else (lambda a: a[ii, ix])
            X, Zv, Pv = [sel(x) for x in xf], sel(Zall), sel(Pall)
            g = alm_wgrads(p, s, X, [sel(l) for l in lf], Zv, Pv, cfg) if cfg.rule == "alm" else pc_wgrads(p, s, X, Zv, Pv, cfg)
            return g, ad_wgrads(p, s, Zv, Pv, cfg)
        f = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
        c = lambda a, b: jnp.dot(f(a), f(b)) / (jnp.linalg.norm(f(a)) * jnp.linalg.norm(f(b)) + 1e-30)
        g, gb = grads(None)
        blocks = [c(g["top"], gb["top"])] + [c(a, b) for a, b in zip(g["mid"], gb["mid"])] + [c(g["bot"], gb["bot"])]
        cs, cu = c(*grads(seen)), c(*grads(uns))
        s_ad = ad_infer(p, s0, Zs, Ps, dataclasses.replace(cfg, c=1.0), T=2000)[0]
        r = hid_residuals(p, s, xf, Pall, cfg.H)
        res = jnp.sqrt(sum(jnp.sum(x ** 2) for x in r) / (sum(jnp.sum(x ** 2) for x in xf) + 1e-30))
        return {"cos_ad": c(g, gb), "cos_ad_seen": cs, "cos_ad_unseen": cu, "cos_ad_blocks": jnp.stack(blocks),
                "ratio_ad": jnp.linalg.norm(f(g)) / (jnp.linalg.norm(f(gb)) + 1e-30),
                "s_dist_ad": jnp.mean(jnp.linalg.norm(s - s_ad, axis=-1) / (jnp.linalg.norm(s_ad, axis=-1) + 1e-8)),
                "hid_res_rel_p2": res}

    prng = np.random.default_rng(999)
    pidx = prng.choice(len(ztr), min(256, len(ztr)), replace=False)
    PZ, PT = jnp.asarray(ztr[pidx]), jnp.asarray(ttr[pidx])
    PPERM = jnp.asarray(np.argsort(prng.random((len(pidx), ztr.shape[1])), 1))

    rng = np.random.default_rng(cfg.seed + 4000)
    log, t0, stop = [], time.time(), (cfg.stop_at or cfg.steps)
    diverged = False
    for i in range(stop):
        idx = rng.integers(0, len(ztr), cfg.batch)
        perm = np.argsort(rng.random((cfg.batch, ztr.shape[1])), 1)
        k = int(rng.integers(1, 4))
        p_new, am_new, st, st_a, m = step(p, amort, st, st_a, jnp.asarray(ztr[idx]), jnp.asarray(ttr[idx]), jnp.asarray(perm), k)
        if not bool(m["finite"]):
            diverged = True
            log.append({"step": i + 1, "diverged": True, **{k_: float(v) for k_, v in m.items()}})
            if verbose:
                print(f"[{cfg.name} c{cfg.c} s{cfg.seed}] DIVERGED at step {i + 1}; keeping the last finite weights", flush=True)
            break
        p, amort = p_new, am_new
        if (i + 1) % log_every == 0 or i == 0 or i == stop - 1:
            row = {"step": i + 1, "k": k, **{k_: (bool(v) if k_ == "finite" else float(v)) for k_, v in m.items()}}
            if cfg.rule in ("pc", "alm") and cfg.log_align:
                for kk in (1, 3):
                    a_ = align(p, amort, PZ, PT, PPERM, kk)
                    row.update({f"{k_}_k{kk}": (np.asarray(v).round(4).tolist() if np.ndim(v) else float(v)) for k_, v in a_.items()})
            if ckpt and (i + 1) in ckpt[1]:
                eqx.tree_serialise_leaves(f"{ckpt[0]}_step{i + 1}.eqx", p)
            log.append(row)
            if verbose:
                print(f"[{cfg.name} c{cfg.c} s{cfg.seed}] " + json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()})
                      + f"  {time.time() - t0:.0f}s", flush=True)
    return p, amort, log, diverged


def make_predict(p, amort, cfg, T=None):
    @eqx.filter_jit
    def _p(p, amort, Zs, Ts, Tt):
        s, _, _, _ = infer_s(p, amort, Zs, Ts, cfg, T)
        return topdown(p, s, phi(Tt), cfg.H)[1]
    return lambda Zs, Ts, Tt: _p(p, amort, jnp.asarray(Zs), jnp.asarray(Ts), jnp.asarray(Tt))


# Finite but blown up (review N3).
# Compares each logged value with BEST value seen BEFORE it (not run median)
#   -- first version used median and flagged every run, 2026-10-13.
# Primary signal: held-out prediction error of batch; secondary: settled energy,
#   with absolute floor since PC's seen-view energy can approach 0.
def exploded(log, factor=5.0, warm=2):
    pv = lambda r: r["pred_err"] / (4 - r["k"]) if "k" in r else r["pred_err"]       # per held-out view
    ev = lambda r: r["E"] / r["k"] if "k" in r else r["E"]                            # per seen view
    P = [pv(r) for r in log if "pred_err" in r and np.isfinite(r["pred_err"])]
    E = [ev(r) for r in log if "E" in r and np.isfinite(r["E"])]
    Rr = [r["res_rel"] for r in log if np.isfinite(r.get("res_rel", np.nan))]
    for i in range(warm, len(P)):
        if P[i] > factor * min(P[:i]):
            return True
    for i in range(warm, len(E)):
        if E[i] > 10.0 * min(E[:i]) + 1.0:
            return True
    for i in range(warm, len(Rr)):          # alm: multiplier loop blowing up (absolute floor: a stalled loop is not a blow-up)
        if Rr[i] > 0.5 and Rr[i] > factor * min(Rr[:i]):
            return True
    return False


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


# Atomic write (review N7); non-finite floats become null (strict JSON).
def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1, default=str)
    os.replace(tmp, path)
