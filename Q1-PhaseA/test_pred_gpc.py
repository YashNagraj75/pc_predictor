"""Gate tests for pred_gpc.py (revision 2).  All must pass before any training run.

    python test_pred_gpc.py                 float32 gates (run on CPU and on the GPU backend)
    JAX_ENABLE_X64=1 python test_pred_gpc.py   float64: PC == BP auto-decoder in the observation-weight -> 0 limit
"""
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


def flat(t):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])


def cos(a, b):
    a, b = flat(a), flat(b)
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def planted(p, cfg, b=32, k=4, noise=0.1, seed=3):
    """Targets a planted generator can explain: Z = G(s_true, tau) + noise."""
    ks, kt, kn = jax.random.split(jax.random.PRNGKey(seed), 3)
    s_true = jax.random.normal(ks, (b, G.M))
    T9 = 0.5 * jax.random.normal(kt, (b, k, 9))
    Z = G.topdown(p, s_true, G.phi(T9), cfg.H)[1] + noise * jax.random.normal(kn, (b, k, 128))
    return Z, T9, s_true


def equivalence():
    assert jnp.zeros(1).dtype == jnp.float64, "run with JAX_ENABLE_X64=1"
    Z, T9 = data(); Ph = G.phi(T9)
    for depth in (3, 10):
        cfg = G.GCfg(depth=depth)
        p = G.init(jax.random.PRNGKey(depth), cfg)
        sf = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (Z.shape[0], G.M))
        Zs = G.topdown(p, sf, Ph, cfg.H)[1] + 0.01 * jax.random.normal(jax.random.PRNGKey(6), Z.shape)
        cb = dataclasses.replace(cfg, T=5000, out_w=1e-4)
        _, xf, _, _ = G.pc_infer(p, sf, Zs, Ph, cb, freeze_s=True)
        c = cos(G.pc_wgrads(p, sf, xf, Zs, Ph, cb), G.ad_wgrads(p, sf, Zs, Ph, cfg))
        check(f"d{depth} (float64) s fixed, observation weight 1e-4: PC weight gradient = BP auto-decoder direction (cos > 0.9999)", c > 0.9999, f"cos {c:.7f}")
        # infinite hidden precision is the auto-decoder limit too (sensitivity arm direction)
        ch = dataclasses.replace(cfg, T=5000, hid_prec=1e4)
        _, xh, _, _ = G.pc_infer(p, sf, Zs, Ph, ch, freeze_s=True)
        c2 = cos(G.pc_wgrads(p, sf, xh, Zs, Ph, ch), G.ad_wgrads(p, sf, Zs, Ph, cfg))   # gradient of the SAME (precision-weighted) energy
        check(f"d{depth} (float64) hidden precision 1e4: relaxed hidden levels = top-down pass, gradient = BP direction (cos > 0.999)", c2 > 0.999, f"cos {c2:.6f}")
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


def main():
    print("matmul precision:", jax.config.jax_default_matmul_precision, "| backend:", jax.default_backend())
    Z, T9 = data()
    Ph = G.phi(T9)
    for depth in (3, 10):
        cfg = G.GCfg(depth=depth, T=300)
        p = G.init(jax.random.PRNGKey(depth), cfg)
        H = cfg.H
        print(f"depth {depth}: H={H} width {cfg.n} params {G.n_params(p)}")
        # power-iteration norm vs exact
        W = p["bot"][0]; ex = float(jnp.linalg.svd(W, compute_uv=False)[0]); pi = float(G._sn(W))
        check(f"d{depth} power-iteration spectral norm is an upper estimate within 5% of exact", ex <= pi <= 1.05 * ex, f"{pi:.4f} vs {ex:.4f}")
        s0 = jnp.zeros((Z.shape[0], G.M))
        for c in G.C_GRID:
            _, _, hist, _ = G.pc_infer(p, s0, Z, Ph, dataclasses.replace(cfg, c=c))
            check(f"d{depth} PC inference stable at c={c} (no increase > 1e-3 of the energy, net decrease)",
                  bool(jnp.all(jnp.diff(hist) <= 1e-3 * jnp.abs(hist[0]))) and float(hist[-1]) < float(hist[0]),
                  f"{float(hist[0]):.1f} -> {float(hist[-1]):.1f}")
        for c in G.C_GRID:
            _, h2, _ = G.ad_infer(p, s0, Z, Ph, dataclasses.replace(cfg, c=c))
            if c > 1.0:
                print(f"     report: BP c={c}: max increase {float(jnp.max(jnp.diff(h2)) / h2[0]):.1e} of the loss, final {float(h2[-1]):.1f}")
                continue
            check(f"d{depth} BP auto-decoder inference stable at c={c} (no increase > 1e-3 of the loss, net decrease)",
                  bool(jnp.all(jnp.diff(h2) <= 1e-3 * jnp.abs(h2[0]))) and float(h2[-1]) < float(h2[0]), f"{float(h2[0]):.1f} -> {float(h2[-1]):.1f}")
        # sensitivity arm: hidden precision 9 stays monotone
        _, _, hh, _ = G.pc_infer(p, s0, Z, Ph, dataclasses.replace(cfg, hid_prec=9.0))
        check(f"d{depth} PC inference stable at hidden precision 9", bool(jnp.all(jnp.diff(hh) <= 1e-3 * jnp.abs(hh[0]))) and float(hh[-1]) < float(hh[0]))
        # convergence on a planted generator with weights at 2x init scale (trained-like), T=400 vs T=4000
        p2 = jax.tree_util.tree_map(lambda x: 2.0 * x, p)
        Zp, Tp, _ = planted(p2, cfg)
        Pp = G.phi(Tp); sz = jnp.zeros((Zp.shape[0], G.M))
        s4 = G.pc_infer(p2, sz, Zp, Pp, dataclasses.replace(cfg, T=400))[0]
        sinf = G.pc_infer(p2, sz, Zp, Pp, dataclasses.replace(cfg, T=4000))[0]
        d4 = float(jnp.linalg.norm(s4 - sinf) / jnp.linalg.norm(sinf))
        print(f"     report: d{depth} planted generator (2x init weights): PC belief at T=400 is {d4:.3f} (relative) from its T=4000 value")
        # locality
        s, xs, _, _ = G.pc_infer(p, s0, Z, Ph, cfg)
        g1 = G.pc_wgrads(p, s, xs, Z, Ph, cfg)
        xs_p = list(xs); xs_p[H - 1] = xs[H - 1] + 1.0
        g2 = G.pc_wgrads(p, s + 1.0, xs_p, Z, Ph, cfg)
        g3 = G.pc_wgrads(p, s, xs, Z + 1.0, Ph, cfg)
        lb, lt = float(jnp.max(jnp.abs(g1["bot"][0] - g2["bot"][0]))), float(jnp.max(jnp.abs(g1["top"][0] - g3["top"][0])))
        check(f"d{depth} PC weight updates are local (bottom ignores s, x_H; top ignores z)", lb == 0.0 and lt == 0.0, f"{lb:.1e} {lt:.1e}")
        # exact fixed point with no prior
        sf = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (Z.shape[0], G.M))
        Ze = G.topdown(p, sf, Ph, H)[1]
        s3, _, h3, _ = G.pc_infer(p, sf, Ze, Ph, dataclasses.replace(cfg, prior=0.0, T=20))
        check(f"d{depth} exact generative fixed point is stationary (prior 0)", float(jnp.max(jnp.abs(s3 - sf))) < 1e-4 and float(h3[-1]) < 1e-6)
        # batch independence
        a4 = G.pc_infer(p, s0[:4], Z[:4], Ph[:4], dataclasses.replace(cfg, T=50))[0]
        a16 = G.pc_infer(p, s0, Z, Ph, dataclasses.replace(cfg, T=50))[0]
        b4 = G.ad_infer(p, s0[:4], Z[:4], Ph[:4], dataclasses.replace(cfg, T=50))[0]
        b16 = G.ad_infer(p, s0, Z, Ph, dataclasses.replace(cfg, T=50))[0]
        bi = max(float(jnp.max(jnp.abs(a4 - a16[:4]))), float(jnp.max(jnp.abs(b4 - b16[:4]))))
        check(f"d{depth} inference independent of the batch (PC and BP)", bi < 1e-5, f"{bi:.1e}")
        perm = jnp.arange(16)[::-1]; inv = jnp.argsort(perm)
        pp = G.pc_infer(p, s0[perm], Z[perm], Ph[perm], dataclasses.replace(cfg, T=50))[0][inv]
        bpp = G.ad_infer(p, s0[perm], Z[perm], Ph[perm], dataclasses.replace(cfg, T=50))[0][inv]
        pi_ = max(float(jnp.max(jnp.abs(pp - a16))), float(jnp.max(jnp.abs(bpp - b16))))
        check(f"d{depth} inference invariant to reordering the batch (PC and BP)", pi_ < 1e-5, f"{pi_:.1e}")
        # prediction depends on the seen views (not vacuous), and T=0 with amortised start is G(s0)
        for rule in ("pc", "bp"):
            cc = dataclasses.replace(cfg, rule=rule, start="amort", T=30)
            am = G.init_amort(jax.random.PRNGKey(9))
            q1 = G.make_predict(p, am, cc)(Z[:, :2], T9[:, :2], T9[:, 2:])
            q2 = G.make_predict(p, am, cc)(Z[:, :2].at[:, 0].add(1.0), T9[:, :2], T9[:, 2:])
            q0 = G.make_predict(p, am, cc, T=0)(Z[:, :2], T9[:, :2], T9[:, 2:])
            s0a = G.start_s(p, am, Z[:, :2], T9[:, :2], cc)
            e0 = float(jnp.max(jnp.abs(q0 - G.topdown(p, s0a, G.phi(T9[:, 2:]), H)[1])))
            check(f"d{depth} {rule}: prediction changes with a seen view; T=0 equals G(amortised s0)", float(jnp.max(jnp.abs(q1 - q2))) > 0 and e0 < 1e-5)
        # predictive training: a few steps per rule run, stay finite, k in {1,2,3} all exercised
        for rule in ("pc", "bp"):
            for start in ("zero", "amort"):
                ct = dataclasses.replace(cfg, rule=rule, start=start, T=20, T2=10, steps=8, batch=16)
                zt = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (64, 4, 128)))
                tt = np.asarray(0.5 * jax.random.normal(jax.random.PRNGKey(2), (64, 4, 9)))
                _, _, log, div = G.train(ct, zt, tt, verbose=False, log_every=1)
                ks = sorted({r["k"] for r in log if "k" in r})
                check(f"d{depth} {rule}/{start}: predictive training runs finite (k seen {ks})", not div and len(log) > 0)
    # divergence is detected and the last finite weights are kept
    cd = G.GCfg(depth=3, c=1e6, T=50, T2=5, steps=5, batch=16)
    zt = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (64, 4, 128))); tt = np.asarray(0.5 * jax.random.normal(jax.random.PRNGKey(2), (64, 4, 9)))
    pd, _, ld, div = G.train(cd, zt, tt, verbose=False, log_every=1)
    check("divergence detected and the returned weights are finite", div and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(pd)))
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if (equivalence() if jnp.zeros(1).dtype == jnp.float64 else main()) else 1)
