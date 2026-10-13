"""Gate tests for pred_gpc.py.  All must pass before any training run."""
import dataclasses, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import pred_gpc as G

OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}")


def data(seed=0, b=16, k=3, d=128):
    kz, kt = jax.random.split(jax.random.PRNGKey(seed + 11))
    return jax.random.normal(kz, (b, k, d)), 0.5 * jax.random.normal(kt, (b, k, 9))


def cos(a, b):
    a = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(a)])
    b = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(b)])
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def equivalence():
    """float64 only (JAX_ENABLE_X64=1): with s fixed and the observation weight -> 0, the PC weight gradient
    at the settled hidden activities equals the BP auto-decoder gradient in direction (Millidge et al. limit).
    float32 cannot resolve this at depth 10 (measured: cos 0.996 at 1e-3, falling with smaller weights)."""
    assert jnp.zeros(1).dtype == jnp.float64, "run with JAX_ENABLE_X64=1"
    Z, T9 = data(); Ph = G.phi(T9)
    for depth in (3, 10):
        cfg = G.GCfg(depth=depth)
        p = G.init(jax.random.PRNGKey(depth), cfg)
        sf = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (Z.shape[0], G.M))
        Zs = G.topdown(p, sf, Ph, cfg.H)[1] + 0.01 * jax.random.normal(jax.random.PRNGKey(6), Z.shape)
        cb = dataclasses.replace(cfg, T=5000, out_w=1e-4)
        _, xf, _ = G.pc_infer(p, sf, Zs, Ph, cb, freeze_s=True)
        c = cos(G.pc_wgrads(p, sf, xf, Zs, Ph, cb), G.ad_wgrads(p, sf, Zs, Ph, cfg))
        check(f"d{depth} (float64) s fixed, observation weight 1e-4: PC weight gradient = BP auto-decoder direction (cos > 0.9999)", c > 0.9999, f"cos {c:.7f}")
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


def main():
    Z, T9 = data()
    Ph = G.phi(T9)
    for depth in (3, 10):
        cfg = G.GCfg(depth=depth, T=300)
        p = G.init(jax.random.PRNGKey(depth), cfg)
        H = cfg.H
        print(f"depth {depth}: H={H} width {cfg.n} params {G.n_params(p)}")
        s0 = jnp.zeros((Z.shape[0], G.M))
        s, xs, hist = G.pc_infer(p, s0, Z, Ph, cfg)
        mono = bool(jnp.all(jnp.diff(hist) <= 1e-5 * jnp.abs(hist[0]) + 1e-6))
        check(f"d{depth} PC inference energy monotone (c=1)", mono, f"{float(hist[0]):.1f} -> {float(hist[-1]):.1f}")
        # Gauss-Newton step: exact curvature only near the generator's range; random Z is far from it, so gate at c=0.5
        s2, h2 = G.ad_infer(p, s0, Z, Ph, dataclasses.replace(cfg, c=0.5))
        check(f"d{depth} BP auto-decoder inference stable (c=0.5): no increase > 1e-3 of the loss, net decrease",
              bool(jnp.all(jnp.diff(h2) <= 1e-3 * jnp.abs(h2[0]))) and float(h2[-1]) < float(h2[0]),
              f"{float(h2[0]):.1f} -> {float(h2[-1]):.1f}")
        _, h1 = G.ad_infer(p, s0, Z, Ph, cfg)
        print(f"     report: c=1 on random targets: {int(jnp.sum(jnp.diff(h1) > 1e-5 * jnp.abs(h1[0])))} of {len(h1) - 1} steps increase the loss")
        # locality: W_1 (bottom) gradient depends only on z and x_1; perturbing x_H and s must not change it
        g1 = G.pc_wgrads(p, s, xs, Z, Ph, cfg)
        xs_p = list(xs); xs_p[H - 1] = xs[H - 1] + 1.0
        g2 = G.pc_wgrads(p, s + 1.0, xs_p, Z, Ph, cfg)
        loc_b = float(jnp.max(jnp.abs(g1["bot"][0] - g2["bot"][0])))
        g3 = G.pc_wgrads(p, s, xs, Z + 1.0, Ph, cfg)
        loc_t = float(jnp.max(jnp.abs(g1["top"][0] - g3["top"][0])))
        check(f"d{depth} PC weight updates are local (bottom ignores s, x_H; top ignores z)", loc_b == 0.0 and loc_t == 0.0, f"{loc_b:.1e} {loc_t:.1e}")
        # small-error equivalence at FIXED s: PC (hidden layers relaxed) vs BP auto-decoder gradient
        sf = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (Z.shape[0], G.M))
        Zs = G.topdown(p, sf, Ph, H)[1] + 0.01 * jax.random.normal(jax.random.PRNGKey(6), Z.shape)
        _, x1, _ = G.pc_infer(p, sf, Zs, Ph, dataclasses.replace(cfg, T=3000), freeze_s=True)
        print(f"     report: at observation weight 1 (training setting) cos = {cos(G.pc_wgrads(p, sf, x1, Zs, Ph, cfg), G.ad_wgrads(p, sf, Zs, Ph, cfg)):.3f} (PC != BP by design)")
        # exact fixed point with no prior: Z = G(s0) -> inference stays put
        c0 = dataclasses.replace(cfg, prior=0.0)
        Ze = G.topdown(p, sf, Ph, H)[1]
        s3, x3, h3 = G.pc_infer(p, sf, Ze, Ph, dataclasses.replace(c0, T=20))
        check(f"d{depth} exact generative fixed point is stationary (prior 0)", float(jnp.max(jnp.abs(s3 - sf))) < 1e-4 and float(h3[-1]) < 1e-6,
              f"|ds| {float(jnp.max(jnp.abs(s3 - sf))):.1e} E {float(h3[-1]):.1e}")
        # batch independence
        a4 = G.pc_infer(p, s0[:4], Z[:4], Ph[:4], dataclasses.replace(cfg, T=50))[0]
        a16 = G.pc_infer(p, s0, Z, Ph, dataclasses.replace(cfg, T=50))[0]
        b4 = G.ad_infer(p, s0[:4], Z[:4], Ph[:4], dataclasses.replace(cfg, T=50))[0]
        b16 = G.ad_infer(p, s0, Z, Ph, dataclasses.replace(cfg, T=50))[0]
        bi = max(float(jnp.max(jnp.abs(a4 - a16[:4]))), float(jnp.max(jnp.abs(b4 - b16[:4]))))
        check(f"d{depth} inference independent of the batch (PC and BP)", bi < 1e-5, f"{bi:.1e}")
        # predict uses only seen views; amortised start runs and both amortiser rules give finite grads
        for rule in ("pc", "bp"):
            cc = dataclasses.replace(cfg, rule=rule, start="amort", T=30)
            am = G.init_amort(jax.random.PRNGKey(9))
            pr = G.make_predict(p, am, cc)
            q1 = pr(Z[:, :2], T9[:, :2], T9[:, 2:])
            q2 = pr(Z.at[:, 2].set(50.0)[:, :2], T9[:, :2], T9[:, 2:])
            ga = G.amort_grads(am, Z, T9, s, cc)
            fin = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(ga))
            check(f"d{depth} {rule}: predict uses only seen views; amortiser grads finite", bool(jnp.all(q1 == q2)) and fin)
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if (equivalence() if jnp.zeros(1).dtype == jnp.float64 else main()) else 1)
