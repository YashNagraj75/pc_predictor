"""Phase P: nonlinear generative PC predictor on FROZEN latents, and its BP auto-decoder control.

Spec: wiki 'Methods/Method - Generative PC predictor plan' (revision 2, after the three-reviewer pass).

Shared belief s (b, m) per image; per view hidden activities x_1..x_H (b, k, n); observation x_0 = z.
Top-down predictions (residual levels at both depths):
    mu_H = W_top [s, phi(tau)] + b_top
    mu_l = x_{l+1} + (1/sqrt(H)) relu(x_{l+1}) W_{l+1}^T          (1 <= l < H)
    mu_0 = relu(x_1) W_1^T + b_0
    E = out_w/2 sum ||z - mu_0||^2 + hid_prec/2 sum_{l=1..H} ||x_l - mu_l||^2 + prior/2 ||s||^2   (summed over views, batch)
Depth = H + 1 weight layers.

PREDICTIVE training (user decision 2026-10-13): each batch draws k in {1,2,3}; per example a random subset of k views is
SEEN.  s* is inferred from the seen views only (T steps); then
  rule "pc": hidden levels of ALL 4 views relaxed with s fixed at s* (T2 steps; seen views warm-started from phase 1),
             weight gradient = dE/dW over all 4 views at those activities (local: each W sees only the levels it joins);
  rule "bp": auto-decoder; s* by GD on the seen-view loss; weights by backprop of the loss on ALL 4 views at s*.
start "zero": s = 0;  "amort" (Fujita): s0 = mean over seen views of A(z_v, tau_v), A regressed onto s* (stop-grad),
           trained by plain PC in PC arms and by backprop in BP arms.
Prediction: infer from the seen views only, then zhat_t = G(s*, tau_t) (top-down pass).
"""
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


@dataclasses.dataclass
class GCfg:
    backbone: str = "bp"
    rule: str = "pc"            # pc | bp
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

    @property
    def H(self):
        return self.depth - 1

    @property
    def n(self):
        return self.width or WIDTH.get(self.depth, 256)

    @property
    def name(self):
        tag = (f"_w{self.width}" if self.width else "") + (f"_hp{self.hid_prec:g}" if self.hid_prec != 1.0 else "")
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


def level_energies(p, s, xs, Z, Ph, H):
    """Unweighted per-level energies [level 0 (observation), 1, ..., H] summed over views and batch."""
    e = [0.5 * jnp.sum((Z - mu_bot(p, xs[0])) ** 2)]
    for l in range(1, H):
        e.append(0.5 * jnp.sum((xs[l - 1] - mu_mid(p["mid"][l - 1], xs[l], H)) ** 2))
    e.append(0.5 * jnp.sum((xs[H - 1] - mu_top(p, s, Ph)) ** 2))
    return e


def energy(p, s, xs, Z, Ph, H, prior, out_w=1.0, hid=1.0):
    e = level_energies(p, s, xs, Z, Ph, H)
    return out_w * e[0] + hid * sum(e[1:]) + 0.5 * prior * jnp.sum(s ** 2)


def ad_loss(p, s, Z, Ph, H, prior):
    return 0.5 * jnp.sum((Z - topdown(p, s, Ph, H)[1]) ** 2) + 0.5 * prior * jnp.sum(s ** 2)


def _E(cfg):
    return lambda p, s, xs, Z, Ph: energy(p, s, xs, Z, Ph, cfg.H, cfg.prior, cfg.out_w, cfg.hid_prec)


# ------------------------------------------------------------------ step sizes
def _sn(W, iters=20):
    """Spectral norm by power iteration from a fixed start (review N5/G9: no SVD in the step), 2% safety margin."""
    W = jax.lax.stop_gradient(W)
    v = jax.random.normal(jax.random.PRNGKey(17), (W.shape[1],))
    v = v / jnp.linalg.norm(v)
    for _ in range(iters):
        v = W.T @ (W @ v)
        v = v / (jnp.linalg.norm(v) + 1e-30)
    return 1.02 * jnp.linalg.norm(W @ v)


def pc_etas(p, k, cfg):
    """[eta_s, eta_x1..eta_xH] = c * 0.9 / (own curvature + 2 * curvature of the term below)."""
    H, h = cfg.H, cfg.hid_prec
    below = [cfg.out_w * _sn(p["bot"][0]) ** 2] + [h * (1.0 + _sn(W) / math.sqrt(H)) ** 2 for W in p["mid"]]
    eta_x = [cfg.c * 0.9 / (h + 2.0 * L) for L in below]
    eta_s = cfg.c * 0.9 / (cfg.prior + 2.0 * h * k * _sn(p["top"][0][:, :M]) ** 2)
    return [eta_s] + eta_x


def gn_lambda(p, s0, Ph, H, iters=25):
    """Per-sample lambda_max(J^T J), J = dG/ds at s0, batched power iteration from a fixed random start."""
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


def pc_infer(p, s0, Z, Ph, cfg, T=None, freeze_s=False, xs0=None):
    """-> s, xs, energy history (T,), size of the next s step relative to |s| (convergence diagnostic)."""
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


def infer_s(p, amort, Z, T9, cfg, T=None):
    """Belief from the given (seen) views: (s*, s0, energy history, relative last step)."""
    Ph = phi(T9)
    s0 = start_s(p, amort, Z, T9, cfg)
    if T == 0:
        return s0, s0, jnp.zeros(1), jnp.zeros(())
    if cfg.rule == "pc":
        s, _, hist, rel = pc_infer(p, s0, Z, Ph, cfg, T)
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
    return PF.bp_grads(amort, x, y, None)


def weight_grads(p, s, Zs_seen_xs, Zall, Pall, seen, cfg):
    """PC phase 2 / BP: gradients on all 4 views at the belief s inferred from the seen views."""
    if cfg.rule == "pc":
        ii = jnp.arange(Zall.shape[0])[:, None]
        xs0 = topdown(p, s, Pall, cfg.H)[0]
        xs0 = [x.at[ii, seen].set(xv) for x, xv in zip(xs0, Zs_seen_xs)]       # warm start the seen views
        _, xs, h2, _ = pc_infer(p, s, Zall, Pall, cfg, T=cfg.T2, freeze_s=True, xs0=xs0)
        return pc_wgrads(p, s, xs, Zall, Pall, cfg), h2
    return ad_wgrads(p, s, Zall, Pall, cfg), jnp.zeros(1)


# ------------------------------------------------------------------ training
def train(cfg, ztr, ttr, verbose=True, log_every=250):
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
        E_init = _E(cfg)(p, s0, topdown(p, s0, Ps, cfg.H)[0], Zs, Ps) if cfg.rule == "pc" else ad_loss(p, s0, Zs, Ps, cfg.H, cfg.prior)
        if cfg.rule == "pc":
            s, xs_seen, hist, rel = pc_infer(p, s0, Zs, Ps, cfg)
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
        up = jnp.mean((jnp.diff(hist) > 1e-3 * jnp.abs(hist[0])).astype(jnp.float32))
        m = {"E_init": E_init / Zs.shape[0], "E": hist[-1] / Zs.shape[0], "pred_err": pred_err, "gnorm": gn,
             "rel_last_step_s": rel, "frac_up_steps": up, "s_norm": jnp.mean(jnp.linalg.norm(s, axis=-1)),
             "amort_rel_err": a_err, "E2": h2[-1] / Zall.shape[0], "finite": finite}
        return p2, amort2, st, st_a2, m

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


def exploded(log, factor=5.0, warm=2):
    """Finite but blown up (review N3).  Compares each logged value with the BEST value seen BEFORE it, so the
    large energy of an untrained model at step 1 is never flagged (the first version compared with the run's median
    and flagged every run, 2026-10-13).  Primary signal: held-out prediction error of the batch (what prediction
    uses); secondary: settled energy, with an absolute floor because PC's seen-view energy can approach 0."""
    P = [r["pred_err"] for r in log if "pred_err" in r and np.isfinite(r["pred_err"])]
    E = [r["E"] for r in log if "E" in r and np.isfinite(r["E"])]
    for i in range(warm, len(P)):
        if P[i] > factor * min(P[:i]):
            return True
    for i in range(warm, len(E)):
        if E[i] > 10.0 * min(E[:i]) + 1.0:
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


def dump_json(obj, path):
    """Atomic write (review N7); non-finite floats become null (strict JSON)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1, default=str)
    os.replace(tmp, path)
