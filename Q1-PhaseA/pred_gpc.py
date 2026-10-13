"""Phase P: nonlinear generative PC predictor on FROZEN latents, and its BP auto-decoder control.

Spec: wiki 'Methods/Method - Generative PC predictor plan'.

Shared belief s (b, m) per image; per seen view hidden activities x_1..x_H (b, k, n); observation x_0 = z (clamped).
Top-down predictions (residual levels so depth is the only architectural variable):
    mu_H = W_top [s, phi(tau)] + b_top
    mu_l = x_{l+1} + (1/sqrt(H)) relu(x_{l+1}) W_{l+1}^T          (1 <= l < H)
    mu_0 = relu(x_1) W_1^T + b_0
    E = sum_v sum_{l=0..H} 1/2 ||x_l - mu_l||^2 + prior/2 ||s||^2       (summed over the batch)
Depth = H + 1 weight layers (3 -> H = 2, 10 -> H = 9).

rule "pc": relax s and every x_l by GD with per-layer curvature steps; weight gradient = dE/dW at the
           settled activities (stop-gradient) -- each W only sees the two levels it connects (local).
rule "bp": auto-decoder.  G(s, tau) = the deterministic top-down pass; infer only s by GD on
           sum_v 1/2||z_v - G(s, tau_v)||^2 + prior/2 ||s||^2; weights by backprop at the inferred s.
start "zero": s = 0;  "amort" = amortised (Fujita): s0 = mean_v A(z_v, tau_v), A a 2-layer MLP regressed onto the
           settled s* (stop-gradient), trained by plain PC in PC arms and by backprop in BP arms.
Prediction: infer from the seen views only, then zhat_t = G(s*, tau_t).
"""
import dataclasses, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx, optax, jpc
import pred_ff as PF

M, DT = 128, 10                 # belief size; phi(tau) size
WIDTH = {3: 600, 10: 256}


@dataclasses.dataclass
class GCfg:
    backbone: str = "bp"
    rule: str = "pc"            # pc | bp
    start: str = "zero"         # zero | amort
    depth: int = 3
    seed: int = 0
    c: float = 1.0              # step-size scale (the only tuned setting)
    T: int = 100
    prior: float = 1.0
    out_w: float = 1.0          # weight on the bottom (observation) term; 1 = standard generative PC.
                                # PC == BP auto-decoder only as out_w -> 0 (test_pred_gpc); training uses 1.
    steps: int = 6000
    batch: int = 256
    lr: float = 1e-3
    wd: float = 1e-4
    T_amort: int = 20           # PC relaxation steps for the amortiser's own (feedforward) PC training

    @property
    def H(self):
        return self.depth - 1

    @property
    def n(self):
        return WIDTH.get(self.depth, 256)

    @property
    def name(self):
        return f"gpc_{self.rule}_d{self.depth}_{self.start}"


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
    """-> xs = [x_1..x_H] (each (b, k, n)) and zhat (b, k, d) by the deterministic top-down pass."""
    xs = [None] * H
    xs[H - 1] = mu_top(p, s, Ph)
    for l in range(H - 1, 0, -1):                 # x_l from x_{l+1}; mid[l-1]
        xs[l - 1] = mu_mid(p["mid"][l - 1], xs[l], H)
    return xs, mu_bot(p, xs[0])


def energy(p, s, xs, Z, Ph, H, prior, out_w=1.0):
    E = 0.5 * out_w * jnp.sum((Z - mu_bot(p, xs[0])) ** 2)
    for l in range(1, H):
        E = E + 0.5 * jnp.sum((xs[l - 1] - mu_mid(p["mid"][l - 1], xs[l], H)) ** 2)
    E = E + 0.5 * jnp.sum((xs[H - 1] - mu_top(p, s, Ph)) ** 2)
    return E + 0.5 * prior * jnp.sum(s ** 2)


def ad_loss(p, s, Z, Ph, H, prior):
    return 0.5 * jnp.sum((Z - topdown(p, s, Ph, H)[1]) ** 2) + 0.5 * prior * jnp.sum(s ** 2)


# ------------------------------------------------------------------ step sizes
def _sn(W):
    return jnp.linalg.norm(jax.lax.stop_gradient(W), 2)


def pc_etas(p, k, cfg):
    """Per-level steps c * 0.9 / (1 + 2 Lambda): [eta_s, eta_x1, ..., eta_xH]."""
    H = cfg.H
    lam_x = [cfg.out_w * _sn(p["bot"][0]) ** 2] + [(1.0 + _sn(W) / math.sqrt(H)) ** 2 for W in p["mid"]]
    lam_s = k * _sn(p["top"][0][:, :M]) ** 2
    f = lambda L, base: cfg.c * 0.9 / (base + 2.0 * L)
    return [f(lam_s, cfg.prior + 1.0)] + [f(L, 1.0) for L in lam_x]


def gn_lambda(p, s0, Ph, H, iters=25):
    """Per-sample largest eigenvalue of J^T J, J = dG(s, tau)/ds at s0 (Gauss-Newton curvature), by batched
    power iteration with a fixed start.  Samples are independent, so this keeps inference batch-independent."""
    p, s0 = jax.lax.stop_gradient(p), jax.lax.stop_gradient(s0)
    f = lambda s: topdown(p, s, Ph, H)[1]
    _, vjp = jax.vjp(f, s0)
    v = jnp.ones_like(s0) / math.sqrt(s0.shape[-1])
    for _ in range(iters):
        w = vjp(jax.jvp(f, (s0,), (v,))[1])[0]
        v = w / (jnp.linalg.norm(w, axis=-1, keepdims=True) + 1e-12)
    w = vjp(jax.jvp(f, (s0,), (v,))[1])[0]
    return jnp.linalg.norm(w, axis=-1)                                    # (b,)


def ad_eta(p, s0, Ph, cfg):
    """Per-sample step c * 0.9 / (prior + 2 lambda_max(J^T J)) -- the same curvature-normalised form as PC's steps."""
    return (cfg.c * 0.9 / (cfg.prior + 2.0 * gn_lambda(p, s0, Ph, cfg.H)))[:, None]


# ------------------------------------------------------------------ inference
def start_s(p, amort, Z, T9, cfg):
    b = Z.shape[0]
    if cfg.start == "zero" or amort is None:
        return jnp.zeros((b, M))
    k = Z.shape[1]
    x = jnp.concatenate([Z, T9], -1).reshape(b * k, -1)
    return PF.forward(amort, x).reshape(b, k, M).mean(1)


def pc_infer(p, s0, Z, Ph, cfg, T=None, freeze_s=False):
    T = cfg.T if T is None else T
    H = cfg.H
    xs0, _ = topdown(p, s0, Ph, H)
    etas = pc_etas(p, Z.shape[1], cfg)
    gfn = jax.grad(lambda v: energy(p, v[0], v[1], Z, Ph, H, cfg.prior, cfg.out_w))

    def step(v, _):
        gs, gx = gfn(v)
        s = v[0] if freeze_s else v[0] - etas[0] * gs
        xs = [x - e * g for x, g, e in zip(v[1], gx, etas[1:])]
        return (s, xs), energy(p, s, xs, Z, Ph, H, cfg.prior, cfg.out_w)

    (s, xs), hist = jax.lax.scan(step, (s0, xs0), None, length=T)
    return s, xs, hist


def ad_infer(p, s0, Z, Ph, cfg, T=None):
    T = cfg.T if T is None else T
    eta = ad_eta(p, s0, Ph, cfg)
    gfn = jax.grad(lambda s: ad_loss(p, s, Z, Ph, cfg.H, cfg.prior))

    def step(s, _):
        s = s - eta * gfn(s)
        return s, ad_loss(p, s, Z, Ph, cfg.H, cfg.prior)

    s, hist = jax.lax.scan(step, s0, None, length=T)
    return s, hist


def infer(p, amort, Z, T9, cfg, T=None):
    Ph = phi(T9)
    s0 = start_s(p, amort, Z, T9, cfg)
    if cfg.rule == "pc":
        s, xs, hist = pc_infer(p, s0, Z, Ph, cfg, T)
    else:
        (s, hist), xs = ad_infer(p, s0, Z, Ph, cfg, T), None
    return s, xs, hist


# ------------------------------------------------------------------ weight gradients
def pc_wgrads(p, s, xs, Z, Ph, cfg):
    s, xs = jax.lax.stop_gradient(s), jax.lax.stop_gradient(xs)
    g = jax.grad(lambda q: energy(q, s, xs, Z, Ph, cfg.H, cfg.prior, cfg.out_w))(p)
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


# ------------------------------------------------------------------ training
def train(cfg, ztr, ttr, verbose=True, log_every=500):
    kp, ka = jax.random.split(jax.random.PRNGKey(cfg.seed))
    p = init(kp, cfg, ztr.shape[-1])
    amort = init_amort(ka, ztr.shape[-1]) if cfg.start == "amort" else None
    mk = lambda: optax.chain(optax.clip_by_global_norm(1.0),
                             optax.adamw(optax.warmup_cosine_decay_schedule(0.0, cfg.lr, min(200, cfg.steps // 4), cfg.steps), weight_decay=cfg.wd))
    opt, opt_a = mk(), mk()
    st = opt.init(p)
    st_a = opt_a.init(eqx.filter(amort, eqx.is_inexact_array)) if amort is not None else None

    @eqx.filter_jit
    def step(p, amort, st, st_a, Z, T9):
        Ph = phi(T9)
        s0 = start_s(p, amort, Z, T9, cfg)
        if cfg.rule == "pc":
            s, xs, hist = pc_infer(p, s0, Z, Ph, cfg)
            g = pc_wgrads(p, s, xs, Z, Ph, cfg)
            fit = 0.5 * jnp.sum((Z - mu_bot(p, xs[0])) ** 2) / Z.shape[0]       # relaxed fit (hidden levels absorb error)
        else:
            s, hist = ad_infer(p, s0, Z, Ph, cfg)
            g = ad_wgrads(p, s, Z, Ph, cfg)
            fit = 0.5 * jnp.sum((Z - topdown(p, s, Ph, cfg.H)[1]) ** 2) / Z.shape[0]
        gen_fit = 0.5 * jnp.sum((Z - topdown(p, jax.lax.stop_gradient(s), Ph, cfg.H)[1]) ** 2) / Z.shape[0]   # what prediction uses
        gn = optax.global_norm(g)
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        a_err = jnp.nan
        if amort is not None:
            ga = amort_grads(amort, Z, T9, s, cfg)
            upd_a, st_a = opt_a.update(ga, st_a, eqx.filter(amort, eqx.is_inexact_array))
            amort = eqx.apply_updates(amort, upd_a)
            a_err = jnp.mean(jnp.sum((s0 - s) ** 2, -1)) / (jnp.mean(jnp.sum(s ** 2, -1)) + 1e-12)
        mono = jnp.all(jnp.diff(hist) <= 1e-4 * jnp.abs(hist[0]) + 1e-6)
        return p, amort, st, st_a, {"E": hist[-1] / Z.shape[0], "E0": hist[0] / Z.shape[0], "fit": fit, "gen_fit": gen_fit, "gnorm": gn,
                                    "monotone": mono, "s_norm": jnp.mean(jnp.linalg.norm(s, axis=-1)), "amort_rel_err": a_err}

    rng = np.random.default_rng(cfg.seed + 4000)
    log, t0 = [], time.time()
    for i in range(cfg.steps):
        idx = rng.integers(0, len(ztr), cfg.batch)
        p, amort, st, st_a, m = step(p, amort, st, st_a, jnp.asarray(ztr[idx]), jnp.asarray(ttr[idx]))
        if (i + 1) % log_every == 0 or i == 0:
            row = {"step": i + 1, **{k: float(v) for k, v in m.items()}}
            if not np.isfinite(row["E"]):
                row["diverged"] = True; log.append(row)
                if verbose:
                    print(f"[{cfg.name} s{cfg.seed}] DIVERGED at {i + 1}", flush=True)
                break
            log.append(row)
            if verbose:
                print(f"[{cfg.name} c{cfg.c} s{cfg.seed}] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
                      + f"  {time.time() - t0:.0f}s", flush=True)
    return p, amort, log


def make_predict(p, amort, cfg, T=None):
    @eqx.filter_jit
    def _p(Zs, Ts, Tt):
        s, _, _ = infer(p, amort, Zs, Ts, cfg, T)
        return topdown(p, s, phi(Tt), cfg.H)[1]
    return lambda Zs, Ts, Tt: _p(jnp.asarray(Zs), jnp.asarray(Ts), jnp.asarray(Tt))
