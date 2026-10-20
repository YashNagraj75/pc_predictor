import argparse, dataclasses, hashlib, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax, jpc
import pred_ff as PF

# Residual feedforward predictor (E1).  Same data, optimiser, steps and evaluation as pred_ff; the network is
# pred_ff's MLP with jpc's identity skips on every interior layer.  BP uses the standard parameterisation (no muPC
# scaling: user decision 2026-10-19, it is a PC modification); PC and PC-ALM use jpc's muPC scaling as the encoder.
WIDTH, ACT = 256, "relu"


@dataclasses.dataclass
class RCfg:
    arm: str = "bp"             # bp | pc | alm
    depth: int = 10             # weight layers (jpc convention); interior layers 1..depth-2 carry skips
    seed: int = 0
    steps: int = 6000
    batch: int = 512
    lr: float = 1e-3
    wd: float = 1e-4
    T: int = 20                 # pc / alm inference steps
    alpha: float = 1.0          # alm dual rate (pc arm forces 0)
    holdout: bool = False       # train on the first 90% of train images, score on the last 10% (selection only)
    backbone: str = "bp"
    layer: str = "emb"
    use_tau: bool = True

    @property
    def ptype(self):
        return "sp" if self.arm == "bp" else "mupc"

    @property
    def gamma(self):
        return 1.0 / math.sqrt(WIDTH) if self.ptype == "mupc" else None    # same output scaling as encoder.gamma_for

    @property
    def name(self):
        return f"ffres_{self.arm}_d{self.depth}"


def make(key, din, dout, cfg):
    return (jpc.make_mlp(key, din, WIDTH, cfg.depth, dout, ACT, use_bias=True, param_type=cfg.ptype),
            jpc.make_skip_model(cfg.depth))


def scalings(params, x, cfg):
    return jpc._get_param_scalings(params[0], x, skip_model=params[1], param_type=cfg.ptype, gamma=cfg.gamma)


def forward(params, x, cfg):
    return jpc.init_activities_with_ffwd(params[0], x, skip_model=params[1], param_type=cfg.ptype, gamma=cfg.gamma)[-1]


def bp_loss(params, x, y, cfg):
    return 0.5 * jnp.sum((y - forward(params, x, cfg)) ** 2) / x.shape[0]


# f_l(prev) for layer l, term for term as jpc.pc_energy_fn (skip added on interior layers only).
def layer_pred(params, l, prev, a):
    model, skip = params
    out = a[l] * jax.vmap(model[l])(prev)
    return out + prev if skip[l] is not None else out


# Hidden residuals r_l = h_l - f_l(h_{l-1}), l = 0..L-2 (h_{-1} = x).
def residuals(params, h, x, a):
    prev, r = x, []
    for l in range(len(h)):
        r.append(h[l] - layer_pred(params, l, prev, a)); prev = h[l]
    return r


# PC-ALM energy, completing-the-square form with rho = 1, output clamped to y, batch mean like jpc.
# lam = None gives the plain PC energy (gated against jpc.pc_energy_fn).
def energy(params, h, x, y, lam, a):
    r = residuals(params, h, x, a)
    if lam is not None:
        r = [ri + li for ri, li in zip(r, lam)]
    out = y - layer_pred(params, len(h), h[-1], a)
    return 0.5 * (sum(jnp.sum(ri ** 2) for ri in r) + jnp.sum(out ** 2)) / x.shape[0]


# Spectral norm by power iteration from a fixed start (no SVD in the training step), 2% safety margin.
def spec_norm(W, iters=20):
    W = jax.lax.stop_gradient(W)
    v = jax.random.normal(jax.random.PRNGKey(17), (W.shape[1],), W.dtype)
    v = v / jnp.linalg.norm(v)
    for _ in range(iters):
        v = W.T @ (W @ v); v = v / (jnp.linalg.norm(v) + 1e-30)
    return 1.02 * jnp.linalg.norm(W @ v)


# Activity step 0.9 / (1 + j)^2, j = max_l ||J_l|| bound (1 + a||W|| through a skip, a||W|| without; ReLU slope <= 1):
# the energy Hessian is M^T M with M = I - (block-subdiagonal J), so ||H|| <= (1 + j)^2.  From the current weights.
def step_size(params, a):
    model, skip = params
    c = []
    for l in range(1, len(model)):
        w = a[l] * spec_norm(model[l].layers[1].weight)
        c.append((1.0 + w) ** 2 if skip[l] is not None else w ** 2)
    return 0.9 / (1.0 + jnp.sqrt(jnp.max(jnp.stack(c)))) ** 2


# T primal steps, each followed by lam += alpha * r (not after the last); weights from the energy at the final state.
# alpha = 0 is plain PC.  Returns the weight gradient (and the final state if asked).
def pc_alm_grads(params, x, y, cfg, return_state=False):
    a = scalings(params, x, cfg)
    alpha = cfg.alpha if cfg.arm == "alm" else 0.0
    h0 = list(jpc.init_activities_with_ffwd(params[0], x, skip_model=params[1], param_type=cfg.ptype, gamma=cfg.gamma)[:-1])
    lam0 = [jnp.zeros_like(v) for v in h0]
    eta, B = step_size(params, a), x.shape[0]
    gfn = jax.grad(lambda h, lam: energy(params, h, x, y, lam, a))

    def step(c, t):
        h, lam = c
        h = [v - eta * B * g for v, g in zip(h, gfn(h, lam))]
        r = residuals(params, h, x, a)
        lam = [jnp.where(t < cfg.T - 1, li + alpha * ri, li) for li, ri in zip(lam, r)]
        return (h, lam), None

    (h, lam), _ = jax.lax.scan(step, (h0, lam0), jnp.arange(cfg.T))
    h, lam = jax.lax.stop_gradient(h), jax.lax.stop_gradient(lam)
    g = eqx.filter_grad(lambda p: energy(p, h, x, y, lam, a))(params)
    return (g, h, lam) if return_state else g


def grads(params, x, y, cfg):
    return eqx.filter_grad(bp_loss)(params, x, y, cfg) if cfg.arm == "bp" else pc_alm_grads(params, x, y, cfg)


def weight_norms(params):
    return [float(spec_norm(m.layers[1].weight)) for m in params[0]]


def make_predict_x(params, cfg):
    f = eqx.filter_jit(lambda p, x: forward(p, x, cfg))
    return lambda X: np.concatenate([np.asarray(f(params, jnp.asarray(X[i:i + 8192]))) for i in range(0, len(X), 8192)])


# Same metrics as pred_ff.evaluate.
def evaluate(params, z, tau, cfg, n_retr=2000):
    X, Y, img, A, Bv = PF.pairs(z, tau, cfg.use_tau)
    P = make_predict_x(params, cfg)(X)
    nmse = float(np.mean(np.sum((P - Y) ** 2, -1)) / np.mean(np.sum(Y ** 2, -1)))
    keep = img < n_retr
    hits = []
    for b in range(4):
        sel = keep & (Bv == b)
        C = z[:n_retr, b]
        d2 = np.sum(P[sel] ** 2, -1)[:, None] - 2 * P[sel] @ C.T + np.sum(C ** 2, -1)[None]
        hits.append(np.argmin(d2, 1) == img[sel])
    return {"nmse": nmse, "retr1": float(np.mean(np.concatenate(hits)))}


def train(cfg, data_path, verbose=True, log_every=500):
    ztr, ttr, zte, tte, _, _ = PF.load(data_path, cfg.backbone, cfg.layer)
    if cfg.holdout:
        nv = int(0.9 * len(ztr)); zte, tte, ztr, ttr = ztr[nv:], ttr[nv:], ztr[:nv], ttr[:nv]
    X, Y, *_ = PF.pairs(ztr, ttr, cfg.use_tau)
    params = make(jax.random.PRNGKey(cfg.seed), X.shape[1], Y.shape[1], cfg)
    opt = PF.optimiser(cfg)
    st = opt.init(eqx.filter(params, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(params, st, x, y):
        g = grads(params, x, y, cfg)
        gn = optax.global_norm(eqx.filter(g, eqx.is_inexact_array))
        upd, st = opt.update(g, st, eqx.filter(params, eqx.is_inexact_array))
        return eqx.apply_updates(params, upd), st, gn, bp_loss(params, x, y, cfg)

    @eqx.filter_jit
    def credit(params, x, y):                 # PC/ALM gradient vs BP's on the SAME network (diagnostic only)
        ga, h, lam = pc_alm_grads(params, x, y, cfg, return_state=True)
        gb = eqx.filter_grad(bp_loss)(params, x, y, cfg)
        f = lambda t: jnp.concatenate([v.ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])
        cs = lambda u, v: jnp.dot(f(u), f(v)) / (jnp.linalg.norm(f(u)) * jnp.linalg.norm(f(v)) + 1e-30)
        r = residuals(params, h, x, scalings(params, x, cfg))
        res = jnp.sqrt(sum(jnp.sum(v ** 2) for v in r) / (sum(jnp.sum(v ** 2) for v in h) + 1e-30))
        return (cs(ga, gb), jnp.linalg.norm(f(ga)) / (jnp.linalg.norm(f(gb)) + 1e-30),
                jnp.stack([cs(u, v) for u, v in zip(ga[0], gb[0])]), res)

    pidx = np.random.default_rng(999).integers(0, len(X), cfg.batch)    # fixed probe batch for the credit readout
    xp, yp = jnp.asarray(X[pidx]), jnp.asarray(Y[pidx])
    rng = np.random.default_rng(cfg.seed + 1000)          # batch stream depends on seed only, as in pred_ff
    log, t0, finite = [], time.time(), True
    for s in range(cfg.steps):
        idx = rng.integers(0, len(X), cfg.batch)
        xb, yb = jnp.asarray(X[idx]), jnp.asarray(Y[idx])
        params_new, st, gn, l = step(params, st, xb, yb)
        if not (np.isfinite(float(l)) and np.isfinite(float(gn))):
            finite = False; log.append({"step": s + 1, "diverged": True}); break
        params = params_new
        if (s + 1) % log_every == 0 or s == 0:
            row = {"step": s + 1, "train_loss": float(l), "grad_norm": float(gn), "w_norms": weight_norms(params)}
            if cfg.arm != "bp":
                c, ratio, lc, res = credit(params, xp, yp)
                row.update({"cos_bp": float(c), "ratio_bp": float(ratio), "layer_cos": np.asarray(lc).round(4).tolist(),
                            "min_layer_cos": float(jnp.min(lc)), "res_rel": float(res)})
            log.append(row)
            if verbose:
                print(f"[{cfg.name} s{cfg.seed}] step {s + 1} loss {float(l):.4f} {row.get('cos_bp', '')} {time.time() - t0:.0f}s", flush=True)
    res = evaluate(params, zte, tte, cfg) if finite else {"nmse": None, "retr1": None}
    if finite and not np.isfinite(res["nmse"]):       # the last update can be the one that broke (review F10)
        finite, res = False, {"nmse": None, "retr1": None}
    return params, log, res, finite


def load_model(path, cfg, din=146, dout=128):
    return eqx.tree_deserialise_leaves(path, make(jax.random.PRNGKey(cfg.seed), din, dout, cfg))


# pred_eval-compatible predict for k = 1 (same contract as pred_ff.make_pair_predict).
def make_pair_predict(params, cfg):
    px = make_predict_x(params, cfg)
    def predict(Zs, Ts, Tt):
        Zs, Ts, Tt = np.asarray(Zs), np.asarray(Ts), np.asarray(Tt)
        assert Zs.shape[1] == 1, "the feedforward predictor takes exactly one seen view"
        m = Tt.shape[1]
        X = np.concatenate([np.repeat(Zs, m, 1), np.repeat(Ts, m, 1), Tt], -1)
        return px(X.reshape(-1, X.shape[-1])).reshape(Zs.shape[0], m, -1)
    return predict


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "highest")     # no TF32, as pred_gpc
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pred_latents.npz"); ap.add_argument("--arm", default="bp")
    ap.add_argument("--depth", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--out", required=True); ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    c = RCfg(arm=a.arm, depth=a.depth, seed=a.seed, steps=a.steps, T=a.T, alpha=a.alpha, holdout=a.holdout)
    if os.path.exists(a.out):
        print("EXISTS -- skipping", a.out); sys.exit(0)
    params, log, res, ok = train(c, a.data)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rec = {"cfg": dataclasses.asdict(c), "ptype": c.ptype, "log": log, "test": res, "finite": ok,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty_files": len(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.splitlines()),
           "matmul_precision": str(jax.config.jax_default_matmul_precision)}
    if ok and a.save:
        eqx.tree_serialise_leaves(a.out.replace(".json", ".eqx"), params)
        rec["eqx_sha256"] = sha256(a.out.replace(".json", ".eqx"))
    json.dump(rec, open(a.out, "w"), indent=1)
    print("TEST", json.dumps(res))
    sys.exit(0 if ok else 3)
