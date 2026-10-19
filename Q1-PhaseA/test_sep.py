import dataclasses as dc, os, sys
X64 = "--x64" in sys.argv
import jax
if X64:
    jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax.numpy as jnp
import pred_gpc as G

# Gates for rule 'sep' in pred_gpc (belief stage, then credit at fixed s).
#   python test_sep.py <depth> [--x64]     one depth per process (compile memory)
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def flat(t):
    return jnp.concatenate([v.ravel() for v in jax.tree_util.tree_leaves(t)])


def cos(a, b):
    a, b = flat(a), flat(b)
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-300))


depth = int([a for a in sys.argv[1:] if not a.startswith("--")][0]) if len(sys.argv) > 1 else 10
dt = jnp.float64 if X64 else jnp.float32
kz, kt = jax.random.split(jax.random.PRNGKey(3))
Z = jax.random.normal(kz, (16, 4, 128), dt); T9 = (0.5 * jax.random.normal(kt, (16, 4, 9))).astype(dt)
Ph = G.phi(T9)
base = G.GCfg(depth=depth, rule="sep", hid_prec=4.0, alpha=1.0, k_inner=1, c=1.0, T=60, T2=60, c_belief=1.0)
p = G.init(jax.random.PRNGKey(depth), base)
p = jax.tree_util.tree_map(lambda v: v.astype(dt), p)
H = base.H
mx = lambda a: float(jnp.max(jnp.abs(a)))

if not X64:
    # 1-2. belief sources reproduce the existing inference exactly
    s0 = jnp.zeros((16, G.M), dt)
    s_pc = G.sep_belief(p, None, Z[:, :2], T9[:, :2], dc.replace(base, belief="pc"))[0]
    ref = G.pc_infer(p, s0, Z[:, :2], Ph[:, :2], dc.replace(base, rule="pc", hid_prec=1.0, c=1.0))[0]
    check(f"d{depth} belief pc == pc_infer at unit precision", mx(s_pc - ref) == 0.0)
    s_or = G.sep_belief(p, None, Z[:, :2], T9[:, :2], dc.replace(base, belief="oracle"))[0]
    ref = G.ad_infer(p, s0, Z[:, :2], Ph[:, :2], dc.replace(base, rule="bp", c=1.0))[0]
    check(f"d{depth} belief oracle == ad_infer", mx(s_or - ref) == 0.0)
    # 3. stage 2 starts at zero hidden residuals
    s = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (16, G.M), dt)
    r0 = G.hid_residuals(p, s, G.topdown(p, s, Ph, H)[0], Ph, H)
    check(f"d{depth} stage 2 starts with every hidden residual zero", max(mx(v) for v in r0) == 0.0)
    # 4. s is held fixed in stage 2, and alpha = 0 leaves the duals at zero
    xs, lam, _ = G.credit_fixed_s(p, s, Z, Ph, dc.replace(base, alpha=0.0))
    check(f"d{depth} alpha = 0: duals stay zero", max(mx(v) for v in lam) == 0.0)
    # 5. amortised belief: prediction uses only the seen views; amortiser and weight grads finite
    ca = dc.replace(base, belief="amort", start="amort")
    am = G.init_amort(jax.random.PRNGKey(9))
    pr = G.make_predict(p, am, ca)
    q1 = pr(Z[:, :2], T9[:, :2], T9[:, 2:]); q2 = pr(Z.at[:, 2].set(50.0)[:, :2], T9[:, :2], T9[:, 2:])
    check(f"d{depth} amort: prediction uses only the seen views", bool(jnp.all(q1 == q2)))
    sa = G.sep_belief(p, am, Z[:, :2], T9[:, :2], ca)[0]
    xs, lam, _ = G.credit_fixed_s(p, sa, Z, Ph, ca)
    tgt = G.sep_s_target(p, sa, xs, lam, Z, Ph, ca)
    ga = G.amort_grads(am, Z[:, :2], T9[:, :2], tgt, ca)
    check(f"d{depth} amort: amortiser grads finite", all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(ga)))
    # 6. every arm trains 6 steps end to end and names are distinct
    zt = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (64, 4, 128)))
    tt = np.asarray(0.5 * jax.random.normal(jax.random.PRNGKey(2), (64, 4, 9)))
    names = set()
    for b in ("oracle", "pc", "amort"):
        for al in (1.0, 0.0):
            for k in (1, 5):
                cc = dc.replace(base, belief=b, start="amort" if b == "amort" else "zero", alpha=al, k_inner=k,
                                T=20, T2=20, steps=6, batch=16)
                p_, _, log, div = G.train(cc, zt, tt, verbose=False, log_every=2)
                names.add(cc.name)
                check(f"d{depth} {cc.name}: 6 steps finite, res_rel logged", (not div) and np.isfinite(log[-1]["pred_err"])
                      and np.isfinite(log[-1]["res_rel"]), f"res_rel {log[-1]['res_rel']:.3g}")
    check(f"d{depth} 12 distinct arm names", len(names) == 12, str(sorted(names)))
else:
    # 7. converged credit at fixed s: small observation error, long stage 2.  The PC-ALM weight gradient should equal
    # the BP auto-decoder's, and the local error at s should equal the auto-decoder's gradient at s.
    s = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (16, G.M), dt)
    Zs = G.topdown(p, s, Ph, H)[1] + 0.01 * jax.random.normal(jax.random.PRNGKey(6), Z.shape, dt)
    gb = G.ad_wgrads(p, s, Zs, Ph, base)
    gs_ad = jax.grad(lambda v: G.ad_loss(p, v, Zs, Ph, H, base.prior))(s)
    for rho, k in ((64.0, 1), (16.0, 1), (16.0, 5), (4.0, 1), (4.0, 5)):
        cc = dc.replace(base, hid_prec=rho, k_inner=k, T2=3000)
        xs, lam, _ = G.credit_fixed_s(p, s, Zs, Ph, cc)
        cw = cos(G.alm_wgrads(p, s, xs, lam, Zs, Ph, cc), gb)
        g_s = jax.grad(lambda v: G.alm_energy(p, v, xs, Zs, Ph, H, cc.prior, cc.out_w, cc.hid_prec, lam))(s)
        cs_ = cos(g_s, gs_ad)
        res = float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in G.hid_residuals(p, s, xs, Ph, H)) / sum(jnp.sum(v ** 2) for v in xs)))
        print(f"     report: rho {rho:g} k_in {k}: cos weight grad {cw:.5f}  cos g_s {cs_:.5f}  hidden residual {res:.2e}")
        if (rho, k) == (64.0, 1):
            check(f"d{depth} float64, rho 64, T2 3000: PC-ALM weight gradient == BP auto-decoder (cos > 0.995)", cw > 0.995, f"{cw:.5f}")
            check(f"d{depth} float64, rho 64, T2 3000: local error at s == auto-decoder gradient (cos > 0.995)", cs_ > 0.995, f"{cs_:.5f}")
print("ALL PASS" if OK[0] else "SOME FAIL")
sys.exit(0 if OK[0] else 1)
