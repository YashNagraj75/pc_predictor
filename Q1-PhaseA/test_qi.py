import dataclasses as dc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import pred_gpc as G

# Gates for the Qi et al. options in pred_gpc (spike, wrule).  The default path must reproduce the pre-change code.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def flat(t):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])


# Pre-change pc_infer and pc_wgrads, copied verbatim (commit 7129350).
def ref_infer(p, s0, Z, Ph, cfg, T, freeze_s=False, xs0=None):
    xs0 = G.topdown(p, s0, Ph, cfg.H)[0] if xs0 is None else xs0
    etas = G.pc_etas(p, Z.shape[1], cfg)
    Ef = G._E(cfg)
    gfn = jax.grad(lambda v: Ef(p, v[0], v[1], Z, Ph))

    def step(v, _):
        gs, gx = gfn(v)
        s = v[0] if freeze_s else v[0] - etas[0] * gs
        xs = [x - e * g for x, g, e in zip(v[1], gx, etas[1:])]
        return (s, xs), Ef(p, s, xs, Z, Ph)
    (s, xs), hist = jax.lax.scan(step, (s0, xs0), None, length=T)
    return s, xs, hist


def ref_wgrads(p, s, xs, Z, Ph, cfg):
    g = jax.grad(lambda q: G._E(cfg)(q, s, xs, Z, Ph))(p)
    return jax.tree_util.tree_map(lambda x: x / Z.shape[0], g)


kz, kt = jax.random.split(jax.random.PRNGKey(3))
Z = jax.random.normal(kz, (16, 3, 128)); T9 = 0.5 * jax.random.normal(kt, (16, 3, 9)); Ph = G.phi(T9)
mx_ = lambda a: float(jnp.max(jnp.abs(a)))
names = set()
DEPTHS = [int(a) for a in sys.argv[1:]] or [3, 10]   # one depth per process keeps compile memory bounded
for depth in DEPTHS:
    jax.clear_caches()
    cfg = G.GCfg(depth=depth, rule="pc", c=2.0); p = G.init(jax.random.PRNGKey(depth), cfg); H = cfg.H
    s0 = jnp.zeros((16, G.M))
    a = G.pc_infer(p, s0, Z, Ph, cfg, T=60); b = ref_infer(p, s0, Z, Ph, cfg, 60)
    d = max(mx_(a[0] - b[0]), max(mx_(x - y) for x, y in zip(a[1], b[1])))
    check(f"d{depth} default pc_infer == pre-change pc_infer", d <= 1e-6 * max(1.0, mx_(b[0])), f"{d:.1e}")
    g1 = G.pc_wgrads(p, a[0], a[1], Z, Ph, cfg); g2 = ref_wgrads(p, a[0], a[1], Z, Ph, cfg)
    check(f"d{depth} default pc_wgrads == pre-change", mx_(flat(g1) - flat(g2)) <= 1e-6 * mx_(flat(g2)))
    rel = mx_(flat(G.fu_grads(p, a[0], a[1], Z, Ph, H, "pc")) - flat(g2)) / mx_(flat(g2))
    check(f"d{depth} fu_grads('pc') == pc_wgrads", rel < 1e-5, f"{rel:.1e}")
    # independent per-layer reference for 'qi' and 'fixed' at a RELAXED state (review N3d)
    s_, xs_ = a[0], a[1]; xtd = G.topdown(p, s_, Ph, H)[0]
    for mode in ("qi", "fixed"):
        g = G.fu_grads(p, s_, xs_, Z, Ph, H, mode)
        pre = xs_ if mode == "qi" else xtd
        ok_ = True
        for l in range(1, H):                       # mid[l-1] maps x_{l+1} (xs[l]) -> x_l (xs[l-1])
            e = xs_[l - 1] - xtd[l - 1]
            ref = -jnp.einsum("bkn,bkm->nm", e, jax.nn.relu(pre[l])) / np.sqrt(H) / Z.shape[0]
            ok_ &= mx_(g["mid"][l - 1] - ref) <= 1e-4 * max(mx_(ref), 1e-12)
        e0 = Z - G.mu_bot(p, xtd[0])
        refb = -jnp.einsum("bkd,bkn->dn", e0, jax.nn.relu(pre[0])) / Z.shape[0]
        ok_ &= mx_(g["bot"][0] - refb) <= 1e-4 * mx_(refb)
        check(f"d{depth} fu_grads('{mode}') == hand-written outer products at a relaxed state", ok_)
    # wavefront: with no spike, after T=t steps levels j >= t and s (t <= H) are still at their initial values
    xs0 = G.topdown(p, s0, Ph, H)[0]
    wf = True
    for t in range(0, H + 1):
        st, xt, _, _ = G.pc_infer(p, s0, Z, Ph, cfg, T=t) if t > 0 else (s0, xs0, None, None)
        wf &= all(mx_(xt[j] - xs0[j]) <= 1e-6 for j in range(t, H)) and mx_(st - s0) <= 1e-6
    st1 = G.pc_infer(p, s0, Z, Ph, cfg, T=H + 1)[0]
    check(f"d{depth} wavefront: level j first moves at t=j, s at t=H (no spike)", wf and mx_(st1 - s0) > 0)
    # lit1: the wavefront step on level j is exactly -grad (step 1); stable: s moves with c_spike at t=H
    cl = dc.replace(cfg, spike="lit1")
    Ef = G._E(cl); ok_ = True
    for j in (0, H // 2, H - 1):
        sj, xj, _, _ = G.pc_infer(p, s0, Z, Ph, cl, T=j) if j > 0 else (s0, xs0, None, None)
        gx = jax.grad(lambda v: Ef(p, v[0], v[1], Z, Ph))((sj, xj))[1]
        xj1 = G.pc_infer(p, s0, Z, Ph, cl, T=j + 1)[1]
        ok_ &= mx_(xj1[j] - (xj[j] - gx[j])) <= 1e-4 * max(mx_(gx[j]), 1e-12)
    check(f"d{depth} lit1: wavefront step on level j is exactly -grad", ok_)
    cs = dc.replace(cfg, spike="stable", c=0.1, c_spike=2.0); cn = dc.replace(cfg, c=0.1)
    # multiplier check from a perturbed state: on the cold start the s gradient at t=H is a near-cancelling residual
    # (float32 round-off differs on GPU); timing itself is covered by the wavefront gate above
    sr = 0.5 * jax.random.normal(jax.random.PRNGKey(11), (16, G.M))
    xr = [x + 0.3 * jax.random.normal(jax.random.PRNGKey(12 + i), x.shape) for i, x in enumerate(G.topdown(p, sr, Ph, H)[0])]
    sH, xH, _, _ = G.pc_infer(p, sr, Z, Ph, cs, T=H, xs0=xr)
    gs = jax.grad(lambda v: G._E(cs)(p, v[0], v[1], Z, Ph))((sH, xH))[0]
    eta_s = G.pc_etas(p, Z.shape[1], cs)[0]
    sH1 = G.pc_infer(p, sr, Z, Ph, cs, T=H + 1, xs0=xr)[0]
    err = mx_(sH1 - (sH - eta_s * 20.0 * gs)) / max(mx_(eta_s * 20.0 * gs), 1e-12)
    errn = mx_(sH1 - (sH - eta_s * gs)) / max(mx_(eta_s * 20.0 * gs), 1e-12)
    check(f"d{depth} stable: s step at t=H uses c_spike/c x base step", err <= 1e-3 and errn > 0.5, f"rel err {err:.1e} (vs no-spike {errn:.2f})")
    # phase 2 never spikes: phase2() of a spike arm == phase2() of the matched no-spike arm on the same inputs
    Pa = Ph; seen = jnp.tile(jnp.arange(1)[None], (16, 1)); xseen = [x[:, :1] for x in a[1]]
    x2a = G.phase2(p, a[0], xseen, Z, Pa, seen, dc.replace(cs, T2=20))[0]
    x2b = G.phase2(p, a[0], xseen, Z, Pa, seen, dc.replace(cn, T2=20))[0]
    check(f"d{depth} phase 2 ignores the spike", max(mx_(u - v) for u, v in zip(x2a, x2b)) == 0.0)
    # every arm: 6 training steps; non-lit1 arms must stay finite; names distinct and carry c
    zt = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (64, 4, 128)))
    tt = np.asarray(0.5 * jax.random.normal(jax.random.PRNGKey(2), (64, 4, 9)))
    arms = (dict(spike="lit1"), dict(spike="stable", c=0.1), dict(c=0.1, tag="_c01"), dict(wrule="qi"),
            dict(wrule="fixed"), dict(c=0.1, wrule="fixed", tag="_c01"), dict(spike="stable", c=0.1, wrule="fixed"),
            dict(spike="stable", c=0.1, wrule="fixed", T=H + 1, T2=H + 1, tag="_qireg"))
    for arm in arms:
        kw = {"T": 20, "T2": 10, "steps": 6, "batch": 16}; kw.update(arm)
        ca = dc.replace(cfg, **kw)
        _, _, log, div = G.train(ca, zt, tt, verbose=False, log_every=2)
        jax.clear_caches()
        fin = (not div) and len(log) > 0 and bool(np.isfinite(log[-1]["pred_err"]))
        print(f"     d{depth} {ca.name}: diverged={div} pred_err={[round(r['pred_err'], 2) for r in log]}")
        names.add(ca.name)
        if ca.spike == "lit1":
            print(f"     report: lit1 finite after 6 steps: {fin}")
        else:
            check(f"d{depth} {ca.name}: finite after 6 training steps", fin)
    check(f"d{depth} new names differ from stage-1 '{cfg.name}'", cfg.name not in names)
check(f"all {8 * len(DEPTHS)} arm names distinct and carry the step scale", len(names) == 8 * len(DEPTHS) and all("_c" in n for n in names), str(sorted(names)))
print("ALL PASS" if OK[0] else "SOME FAIL")
sys.exit(0 if OK[0] else 1)
