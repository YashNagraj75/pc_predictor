import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import pred_sdpc as S

OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}")


def small(seed=0, b=16, k=3, d=12, sizes=(16, 12, 10, 8, 6)):
    kz, kt = jax.random.split(jax.random.PRNGKey(seed + 7))
    Ds = S.init(jax.random.PRNGKey(seed), d, sizes)
    Z = jax.random.normal(kz, (b, k, d)); T = 0.5 * jax.random.normal(kt, (b, k, 9))
    return Ds, Z, T


def main():
    Ds, Z, T = small()
    Ph = S.phi(T)
    cfg = S.SCfg(beta=1.0, lam=0.05, T_inf=3000)
    codes, hist = S.infer(Ds, Z, Ph, cfg, return_hist=True)
    mono = bool(jnp.all(jnp.diff(hist) <= 1e-5 * jnp.abs(hist[0]) + 1e-7))
    check("inference energy (smooth + L1) is monotone non-increasing", mono, f"{float(hist[0]):.4f} -> {float(hist[-1]):.4f}")
    # prox fixed point: a = max(0, a - eta (grad + lam))
    g = jax.grad(lambda c: S.smooth_energy(Ds, c, Z, Ph, cfg.beta))(codes)
    etas = S.steps_per_layer(Ds, Ph, cfg.beta)
    res = max(float(jnp.max(jnp.abs(c - jnp.maximum(0, c - e[:, None] * (gc + cfg.lam))) / e[:, None]))
              for c, gc, e in zip(codes, g, etas))
    check("fixed point satisfies the prox (KKT) condition, residual < 1e-3", res < 1e-3, f"{res:.2e}")
    gl, ga = S.local_dict_grads(Ds, codes, Z, Ph, cfg), S.dict_grads(Ds, codes, Z, Ph, cfg)
    err = max(float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-12)) for a, b in zip(gl, ga))
    check("local outer-product dictionary update equals autodiff dE/dD at fixed codes", err < 1e-5, f"{err:.1e}")
    Dn = S.normalise([d * 3.0 for d in Ds])
    un = max([float(jnp.max(jnp.abs(jnp.linalg.norm(Dn[0], axis=0) - 1)))] +
             [float(jnp.max(jnp.abs(jnp.linalg.norm(D, axis=0) - 1))) for D in Dn[1:]])
    check("unit-norm projection of every dictionary column", un < 1e-5, f"{un:.1e}")
    c4 = S.infer(Ds, Z[:4], Ph[:4], S.SCfg(T_inf=200))
    c16 = S.infer(Ds, Z, Ph, S.SCfg(T_inf=200))
    bi = max(float(jnp.max(jnp.abs(a[:4] - b))) for a, b in zip(c16, c4))
    check("per-image inference independent of the batch", bi < 1e-5, f"{bi:.1e}")
    # one layer, lam = 0, signed least squares: compare the bottom fit with lstsq (non-negativity aside -> use positive target)
    D1 = Ds[0][:, :4]                                        # (d, 4, 10): small, well-conditioned
    zt = S.bottom(D1, jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (16, 4))), Ph)   # realisable with a >= 0
    one = [D1]
    c1 = S.infer(one, zt, Ph, S.SCfg(lam=0.0, T_inf=20000))
    fit = float(jnp.linalg.norm(S.bottom(D1, c1[0], Ph) - zt) / jnp.linalg.norm(zt))
    check("one layer, lam = 0: inference recovers an exactly realisable non-negative code (rel fit < 1e-3)", fit < 1e-3, f"{fit:.1e}")
    pred = S.make_predict(Ds, S.SCfg(T_inf=50))
    Zs, Ts, Tt = Z[:, :2], T[:, :2], T[:, 2:]
    p1 = pred(Zs, Ts, Tt)
    p2 = pred(Zs, Ts, Tt + 0.0)
    Zalt = Z.at[:, 2].set(100.0)
    p3 = pred(Zalt[:, :2], Ts, Tt)
    check("predict() uses only the seen views (unseen latent changed -> identical prediction)",
          bool(jnp.all(p1 == p3)) and bool(jnp.all(p1 == p2)))
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
