"""Phase P: pure SDPC-style generative predictor on FROZEN latents (5 sparse layers, local learning).

Per image a hierarchy of non-negative sparse codes a_1..a_5 (a_5 = top belief), sizes SIZES.
Energy over the OBSERVED views S of one image:
    E = sum_{v in S} 1/2 ||z_v - D_1(a_1 (x) phi(tau_v))||^2 + sum_{l=2..5} beta/2 ||a_{l-1} - D_l a_l||^2
        + sum_l lam_l ||a_l||_1 ,      phi(tau) = [tau, 1]  (10-d),   D_1 : (d, n_1, 10)  bilinear.
Inference: ISTA on all codes jointly from zero, prox = max(0, a - eta_l lam_l) (non-negative, SDPC);
per-layer step eta_l = 0.9 / G_l, G_l a Gershgorin bound on the block row of the code Hessian
(diagonal block norm + coupling norms), from the current dictionaries (Fujita-style per-layer step).
Learning: each dictionary gets the gradient of E at the FIXED inferred codes (local and Hebbian:
dE/dD_1 = -sum_v eps_v (a_1 (x) phi_v)^T, dE/dD_l = -beta (a_{l-1} - D_l a_l) a_l^T), averaged over the
batch, AdamW, then unit-norm dictionary columns.  Nothing backpropagates through inference.
Prediction for a target view t: zhat_t = D_1(a_1* (x) phi(tau_t)) with a* inferred from the seen views only.
"""
import argparse, dataclasses, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp, optax

SIZES = (256, 256, 256, 256, 128)


@dataclasses.dataclass
class SCfg:
    backbone: str = "bp"
    seed: int = 0
    beta: float = 1.0
    lam: float = 0.05            # lam_l = lam * lam_decay**(l-1)
    lam_decay: float = 1.0
    T_inf: int = 100
    steps: int = 4000
    batch: int = 256
    lr: float = 1e-3
    wd: float = 0.0


def phi(tau):
    return jnp.concatenate([tau, jnp.ones(tau.shape[:-1] + (1,), tau.dtype)], -1)


def init(key, d, sizes=SIZES):
    ks = jax.random.split(key, len(sizes))
    D1 = jax.random.normal(ks[0], (d, sizes[0], 10))
    Ds = [D1] + [jax.random.normal(ks[l], (sizes[l - 1], sizes[l])) for l in range(1, len(sizes))]
    return normalise(Ds)


def normalise(Ds):
    D1 = Ds[0] / (jnp.linalg.norm(Ds[0], axis=0, keepdims=True) + 1e-12)      # unit (d,) column per (k, p)
    return [D1] + [D / (jnp.linalg.norm(D, axis=0, keepdims=True) + 1e-12) for D in Ds[1:]]


def bottom(D1, a1, ph):
    """a1 (b, n1), ph (b, m, 10) -> (b, m, d)."""
    return jnp.einsum("dkp,bk,bmp->bmd", D1, a1, ph)


def smooth_energy(Ds, codes, Z, Ph, beta, mask=None):
    """Per-image smooth energy, summed over the batch.  Z (b, k, d), Ph (b, k, 10); mask (b, k) of observed views."""
    eps = Z - bottom(Ds[0], codes[0], Ph)
    e = 0.5 * jnp.sum(eps ** 2, -1)
    if mask is not None:
        e = e * mask
    E = jnp.sum(e)
    for l in range(1, len(Ds)):
        E = E + 0.5 * beta * jnp.sum((codes[l - 1] - codes[l] @ Ds[l].T) ** 2)
    return E


def steps_per_layer(Ds, Ph, beta, mask=None):
    """Gershgorin bound G_l per image for layer l (b,) or scalar; step 0.9 / G_l."""
    M = jnp.einsum("dkp,bmp->bmdk", Ds[0], Ph)                          # (b, m, d, n1) per-view bottom maps
    sq = jnp.sum(M ** 2, axis=(2, 3))                                    # Frobenius^2 >= spectral^2 (cheap, safe)
    if mask is not None:
        sq = sq * mask
    L1 = jnp.sum(sq, -1)                                                 # (b,)
    nrm = [jnp.linalg.norm(D, 2) for D in Ds[1:]]                        # ||D_l||_2, l = 2..5
    if len(Ds) == 1:
        return [0.9 / L1]
    G = [L1 + beta + beta * nrm[0]]                                      # a_1: bottom + its own prior + coupling to a_2
    for l in range(1, len(Ds)):
        g = beta * nrm[l - 1] ** 2 + beta * nrm[l - 1]                   # a_l predicts a_{l-1}
        if l + 1 < len(Ds):
            g = g + beta + beta * nrm[l]                                 # a_l is predicted by a_{l+1}
        G.append(g * jnp.ones_like(L1))
    return [0.9 / g for g in G]


def infer(Ds, Z, Ph, cfg, mask=None, T=None, return_hist=False):
    T = cfg.T_inf if T is None else T
    b = Z.shape[0]
    codes = [jnp.zeros((b, n)) for n in [Ds[0].shape[1]] + [D.shape[1] for D in Ds[1:]]]
    etas = steps_per_layer(Ds, Ph, cfg.beta, mask)
    lams = [cfg.lam * cfg.lam_decay ** l for l in range(len(Ds))]
    gfn = jax.grad(lambda c: smooth_energy(Ds, c, Z, Ph, cfg.beta, mask))

    def total(c):
        return smooth_energy(Ds, c, Z, Ph, cfg.beta, mask) + sum(lm * jnp.sum(jnp.abs(x)) for lm, x in zip(lams, c))

    def step(c, _):
        g = gfn(c)
        c = [jnp.maximum(0.0, x - e[:, None] * gx - e[:, None] * lm) for x, gx, e, lm in zip(c, g, etas, lams)]
        return c, total(c)

    codes, hist = jax.lax.scan(step, codes, None, length=T)
    return (codes, hist) if return_hist else codes


def dict_grads(Ds, codes, Z, Ph, cfg, mask=None):
    """Gradient of E w.r.t. the dictionaries at FIXED codes, batch mean."""
    return jax.grad(lambda D: smooth_energy(D, codes, Z, Ph, cfg.beta, mask))(Ds)


def local_dict_grads(Ds, codes, Z, Ph, cfg):
    """The same, written as explicit local outer products (gate test target)."""
    eps = Z - bottom(Ds[0], codes[0], Ph)                                # (b, k, d)
    g = [-jnp.einsum("bmd,bk,bmp->dkp", eps, codes[0], Ph)]
    for l in range(1, len(Ds)):
        r = codes[l - 1] - codes[l] @ Ds[l].T
        g.append(-cfg.beta * r.T @ codes[l])
    return g


def make_predict(Ds, cfg, T=None):
    @jax.jit
    def _p(Zs, Ts, Tt):
        codes = infer(Ds, Zs, phi(Ts), cfg, T=T)
        return bottom(Ds[0], codes[0], phi(Tt))
    return lambda Zs, Ts, Tt: _p(jnp.asarray(Zs), jnp.asarray(Ts), jnp.asarray(Tt))


def train(cfg, ztr, ttr, verbose=True, eval_fn=None):
    Ds = init(jax.random.PRNGKey(cfg.seed), ztr.shape[-1])
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, cfg.lr, 200, cfg.steps), weight_decay=cfg.wd)
    st = opt.init(Ds)

    @jax.jit
    def step(Ds, st, Z, T):
        Ph = phi(T)
        codes, hist = infer(Ds, Z, Ph, cfg, return_hist=True)
        g = [x / Z.shape[0] for x in dict_grads(Ds, codes, Z, Ph, cfg)]
        upd, st = opt.update(g, st, Ds)
        Ds = normalise(optax.apply_updates(Ds, upd))
        act = [jnp.mean(c > 0) for c in codes]
        return Ds, st, hist[-1] / Z.shape[0], act, jnp.all(jnp.diff(hist) <= 1e-4 * jnp.abs(hist[0]) + 1e-6)

    rng = np.random.default_rng(cfg.seed + 2000)
    log, t0 = [], time.time()
    for s in range(cfg.steps):
        idx = rng.integers(0, len(ztr), cfg.batch)
        Ds, st, E, act, mono = step(Ds, st, jnp.asarray(ztr[idx]), jnp.asarray(ttr[idx]))
        if (s + 1) % 500 == 0 or s == 0:
            row = {"step": s + 1, "energy": float(E), "active": [round(float(a), 4) for a in act], "monotone": bool(mono)}
            if eval_fn is not None and (s + 1) % 1000 == 0:
                row["val"] = eval_fn(Ds)
            log.append(row)
            if verbose:
                print(f"[sdpc {cfg.backbone} s{cfg.seed}] {json.dumps(row)}  {time.time() - t0:.0f}s", flush=True)
    return Ds, log


def n_params(Ds):
    return int(sum(np.prod(D.shape) for D in Ds))
