import dataclasses as dc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import pred_ffres as R, pred_gpc as G, diag_epc as E, diag_e1_credit as DC

# Gates for diag_epc.py: known answers for the ePC port, on random weights (fast) and on one trained checkpoint each.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def rel(a, b):
    a, b = E.flat(a), E.flat(b)
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


def ff_random(depth=3, B=64, seed=0, act=None):
    cfg = R.RCfg(arm="pc", depth=depth, seed=seed)
    params = R.make(jax.random.PRNGKey(seed), 146, 128, cfg)
    if act is not None:                          # same architecture, other activation (linear: convex energy)
        import jpc
        params = (jpc.make_mlp(jax.random.PRNGKey(seed), 146, R.WIDTH, depth, 128, act, use_bias=True, param_type=cfg.ptype), params[1])
    kx, ky = jax.random.split(jax.random.PRNGKey(seed + 7))
    return cfg, params, jax.random.normal(kx, (B, 146)), 0.5 * jax.random.normal(ky, (B, 128))


def main():
    # 1. ff: power iteration == exact largest eigenvalue of J^T J (per sample)
    cfg, params, x, y = ff_random()
    a = R.scalings(params, x, cfg); h0 = R.activities(params, x, cfg)[:-1]; e0 = [jnp.zeros_like(v) for v in h0]
    yhat = lambda e: R.layer_pred(params, len(e), E.ff_states(params, e, x, a)[-1], a)
    lam = E.gn_max(yhat, e0, iters=60)
    exact = []
    for i in range(4):
        f1 = lambda flat_e: yhat([v.at[i].set(fe) for v, fe in zip(e0, jnp.split(flat_e, np.cumsum([v.shape[1] for v in e0])[:-1]))])[i]
        J = jax.jacrev(f1)(jnp.zeros(sum(v.shape[1] for v in e0)))
        exact.append(float(jnp.linalg.eigvalsh(J.T @ J)[-1]))
    err = max(abs(float(lam[i]) - exact[i]) / exact[i] for i in range(4))
    check("ff: power-iteration lambda_max(J^T J) == exact, per sample", err < 1e-2, f"max rel err {err:.1e}")
    # 2. ff: states from errors; error energy == state energy (their E vs E_local)
    g, h, u, e = E.ff_epc(params, x, y, cfg, 50, 1.0, 0.0)
    Ee = 0.5 * (sum(jnp.sum(v ** 2) for v in e) + jnp.sum((y - R.layer_pred(params, len(h), h[-1], a)) ** 2))
    Es = R.energy(params, h, x, y, None, a, 1.0) * x.shape[0]
    check("ff: error-based energy == state-based energy at the ePC state", abs(float(Ee - Es)) / float(Es) < 1e-5, f"{float(Ee):.5f} vs {float(Es):.5f}")
    # 3. ff: same equilibrium as sPC on a LINEAR d3 network (convex energy, unique minimum).  With ReLU both methods
    #    chatter at kinks and stop at different non-stationary points (dbg c43fdb0b), so equality is only defined here.
    cl, pl, xl, yl = ff_random(act="linear")
    ge, he, *_ = E.ff_epc(pl, xl, yl, cl, 4000, 1.0, 0.0)
    gs, hs, *_ = DC.run(pl, xl, yl, cl, 40000, 1.0, 0.0)
    al = R.scalings(pl, xl, cl); Eh = jax.grad(lambda h: R.energy(pl, h, xl, yl, None, al, 1.0))
    st = E.flat(Eh(he)); st0 = E.flat(Eh(R.activities(pl, xl, cl)[:-1]))
    check("ff linear d3: ePC and sPC reach the same minimum (state, weight gradient) and ePC is stationary",
          rel(he, hs) < 1e-3 and E.cos(ge, gs) > 0.9999 and float(jnp.linalg.norm(st) / jnp.linalg.norm(st0)) < 1e-4,
          f"state rel {rel(he, hs):.1e}, grad cos {E.cos(ge, gs):.6f}, stationarity {float(jnp.linalg.norm(st) / jnp.linalg.norm(st0)):.1e}")
    # 4. ff: energy non-increasing (alpha 0), and large-rho limit == BP, on a trained d10 checkpoint
    ck = E.ff_ckpts()
    check("ff: trained depth-10 checkpoints found", len(ck) >= 1, str([n for n, *_ in ck]))
    if ck:
        name, cfg10, p10 = ck[0]
        from diag_e1_credit import flat as _f
        import pred_ff as PF
        ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb"); X, Y, *_ = PF.pairs(ztr, ttr, True)
        idx = np.random.default_rng(999).integers(0, len(X), 512); xx, yy = jnp.asarray(X[idx]), jnp.asarray(Y[idx])
        *_, hist = E.ff_epc(p10, xx, yy, cfg10, 400, 1.0, 0.0, return_hist=True); hh = np.asarray(hist)
        print(f"     report: {name} ePC rho 1 T 400: energy {hh[0]:.2f} -> {hh[-1]:.2f}, rising steps {np.mean(np.diff(hh) > 0):.3f}, "
              f"max rise rel {np.max(np.diff(hh) / np.abs(hh[:-1])):.1e} (ReLU kinks)")
        check(f"ff {name}: ePC lowers the energy overall (T 400)", hh[-1] < hh[0])
        gb = eqx.filter_grad(R.bp_loss)(p10, xx, yy, cfg10)
        gL = E.ff_epc(p10, xx, yy, cfg10, 400, 1e4, 0.0)[0]
        check(f"ff {name}: ePC at rho 1e4 == BP gradient (cos > 0.999, ratio within 2%)", E.cos(gL, gb) > 0.999 and abs(E.ratio(gL, gb) - 1) < 0.02,
              f"cos {E.cos(gL, gb):.5f} ratio {E.ratio(gL, gb):.4f}")
    # 5. gen: zero errors == topdown; error energy == pred_gpc energy
    gcfg = G.GCfg(depth=3); gp = G.init(jax.random.PRNGKey(3), gcfg)
    Z = jax.random.normal(jax.random.PRNGKey(4), (16, 3, 128)); Ph = G.phi(0.5 * jax.random.normal(jax.random.PRNGKey(5), (16, 3, 9)))
    s = 0.5 * jax.random.normal(jax.random.PRNGKey(6), (16, G.M))
    xs0 = G.topdown(gp, s, Ph, gcfg.H)[0]; ez = [jnp.zeros_like(v) for v in xs0]
    check("gen: zero errors reproduce the top-down pass", rel(E.gen_states(gp, s, ez, Ph, gcfg.H), xs0) < 1e-6)
    er = [0.1 * jax.random.normal(jax.random.PRNGKey(10 + i), v.shape) for i, v in enumerate(xs0)]
    xs = E.gen_states(gp, s, er, Ph, gcfg.H); c4 = dc.replace(gcfg, hid_prec=4.0)
    Eg = G.energy(gp, s, xs, Z, Ph, gcfg.H, 1.0, 1.0, 4.0)
    Eh = 4.0 * 0.5 * sum(jnp.sum(v ** 2) for v in er) + 0.5 * jnp.sum((Z - G.mu_bot(gp, xs[0])) ** 2) + 0.5 * jnp.sum(s ** 2)
    check("gen: pred_gpc energy == error-based energy", abs(float(Eg - Eh)) / float(Eh) < 1e-5)
    # 6-7. gen, LINEAR d3 (relu patched to identity: convex energy).  (6) ePC == sPC minimum; (7) at hidden precision
    #      1e3 x the decoder's gain lambda_e, ePC == the BP auto-decoder (belief and weight gradient).  On the trained
    #      ReLU d10 decoder 1e4 is NOT large (gain is higher; dbg c43fdb0b): that is a result, not a gate.
    relu = jax.nn.relu
    try:
        jax.nn.relu = lambda v: v
        gl = G.init(jax.random.PRNGKey(3), gcfg); s0 = jnp.zeros((16, G.M))
        se, xe, _ = E.gen_epc(gl, Z, Ph, gcfg, 6000)
        ss, xs_, _, _ = G.pc_infer(gl, s0, Z, Ph, gcfg, 60000, spike_on=False)
        Ev = jax.grad(lambda v: G._E(gcfg)(gl, v[0], v[1], Z, Ph))
        stat = float(jnp.linalg.norm(E.flat(Ev((se, xe)))) / jnp.linalg.norm(E.flat(Ev((s0, G.topdown(gl, s0, Ph, gcfg.H)[0])))))
        check("gen linear d3: ePC and sPC reach the same minimum, ePC stationary", rel(se, ss) < 1e-3 and rel(xe, xs_) < 1e-3 and stat < 1e-4,
              f"s rel {rel(se, ss):.1e}, x rel {rel(xe, xs_):.1e}, stationarity {stat:.1e}")
        e0 = [jnp.zeros_like(v) for v in G.topdown(gl, s0, Ph, gcfg.H)[0]]
        lam_e = float(jnp.max(E.gn_max(lambda e_: G.mu_bot(gl, E.gen_states(gl, s0, e_, Ph, gcfg.H)[0]), e0)))
        cad = dc.replace(gcfg, c=0.25)
        s_ad = G.ad_infer(gl, s0, Z, Ph, cad, 40000)[0]; g_ad = G.ad_wgrads(gl, s_ad, Z, Ph, cad)
        cL = dc.replace(gcfg, hid_prec=1e3 * lam_e)
        sL, xL, _ = E.gen_epc(gl, Z, Ph, cL, 20000)
        m = E.gen_metrics(gl, sL, xL, Z, Ph, cL, s_ad, g_ad)
        check(f"gen linear d3: ePC at hidden precision 1e3 x gain ({1e3 * lam_e:.0f}) == BP auto-decoder", m["s_dist_ad"] < 0.01 and m["cos"] > 0.999 and abs(m["ratio"] - 1) < 0.02,
              f"s_dist {m['s_dist_ad']:.1e} cos {m['cos']:.5f} ratio {m['ratio']:.4f}")
    finally:
        jax.nn.relu = relu
    # report: the trained d10 decoder's gain, which sets how large a precision counts as large there
    gk = E.gen_ckpts()
    check("gen: trained depth-10 checkpoints found", len(gk) >= 1, str([n for n, *_ in gk]))
    for name, c0, p in gk[:1]:
        Zb, Phb = E.gen_batch(64); sz = jnp.zeros((64, G.M))
        e0 = [jnp.zeros_like(v) for v in G.topdown(p, sz, Phb, c0.H)[0]]
        le = E.gn_max(lambda e_: G.mu_bot(p, E.gen_states(p, sz, e_, Phb, c0.H)[0]), e0)
        print(f"     report: {name} decoder gain lambda_max(J_e^T J_e): median {float(jnp.median(le)):.3g}, max {float(jnp.max(le)):.3g}")
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
