"""Backprop SET predictor (baseline for the generative predictors): any number of seen views.

    zhat_t = g( mean_{v in S} h(z_v, tau_v), tau_t ),   h, g MLPs (ReLU, width 256), trained by backprop.
Training: each batch draws k in {1,2,3}; per image a random view permutation, first k seen, the other 4-k are targets.
"""
import dataclasses, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx, optax


@dataclasses.dataclass
class BCfg:
    seed: int = 0
    steps: int = 6000
    batch: int = 256
    lr: float = 1e-3
    wd: float = 1e-4
    width: int = 256


class SetPred(eqx.Module):
    h: eqx.nn.MLP
    g: eqx.nn.MLP

    def __init__(self, key, d=128, dt=9, w=256):
        k1, k2 = jax.random.split(key)
        self.h = eqx.nn.MLP(d + dt, w, w, 1, activation=jax.nn.relu, key=k1)
        self.g = eqx.nn.MLP(w + dt, d, w, 2, activation=jax.nn.relu, key=k2)

    def __call__(self, Zs, Ts, Tt):
        """Zs (k, d), Ts (k, 9), Tt (m, 9) -> (m, d)"""
        pool = jnp.mean(jax.vmap(self.h)(jnp.concatenate([Zs, Ts], -1)), 0)
        return jax.vmap(lambda t: self.g(jnp.concatenate([pool, t])))(Tt)


def train(cfg, ztr, ttr, verbose=True):
    model = SetPred(jax.random.PRNGKey(cfg.seed), ztr.shape[-1], ttr.shape[-1], cfg.width)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(optax.warmup_cosine_decay_schedule(0.0, cfg.lr, 200, cfg.steps), weight_decay=cfg.wd))
    st = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def loss(m, Zs, Ts, Zt, Tt):
        P = jax.vmap(m)(Zs, Ts, Tt)
        return 0.5 * jnp.mean(jnp.sum((P - Zt) ** 2, -1))

    @eqx.filter_jit
    def step(m, st, Zs, Ts, Zt, Tt):
        l, g = eqx.filter_value_and_grad(loss)(m, Zs, Ts, Zt, Tt)
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, l

    rng = np.random.default_rng(cfg.seed + 3000)
    t0 = time.time()
    for s in range(cfg.steps):
        k = int(rng.integers(1, 4))
        idx = rng.integers(0, len(ztr), cfg.batch)
        perm = np.argsort(rng.random((cfg.batch, 4)), 1)
        ii = idx[:, None]
        model, st, l = step(model, st, jnp.asarray(ztr[ii, perm[:, :k]]), jnp.asarray(ttr[ii, perm[:, :k]]),
                            jnp.asarray(ztr[ii, perm[:, k:]]), jnp.asarray(ttr[ii, perm[:, k:]]))
        if verbose and (s + 1) % 2000 == 0:
            print(f"[bpset s{cfg.seed}] step {s + 1} loss {float(l):.3f} {time.time() - t0:.0f}s", flush=True)
    return model


def make_predict(model):
    f = eqx.filter_jit(lambda Zs, Ts, Tt: jax.vmap(model)(Zs, Ts, Tt))
    return lambda Zs, Ts, Tt: f(jnp.asarray(Zs), jnp.asarray(Ts), jnp.asarray(Tt))


def n_params(model):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))))
