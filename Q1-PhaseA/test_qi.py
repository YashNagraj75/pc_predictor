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
names = set()
for depth in (3, 10):
    cfg = G.GCfg(depth=depth, rule="pc", c=2.0); p = G.init(jax.random.PRNGKey(depth), cfg); H = cfg.H
    s0 = jnp.zeros((16, G.M))
    a = G.pc_infer(p, s0, Z, Ph, cfg, T=60); b = ref_infer(p, s0, Z, Ph, cfg, 60)
    d = max(float(jnp.max(jnp.abs(a[0] - b[0]))), max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a[1], b[1])), float(jnp.max(jnp.abs(a[2] - b[2]))))
    check(f"d{depth} default pc_infer == pre-change pc_infer", d == 0.0, f"{d:.1e}")
    g1 = G.pc_wgrads(p, a[0], a[1], Z, Ph, cfg); g2 = ref_wgrads(p, a[0], a[1], Z, Ph, cfg)
    check(f"d{depth} default pc_wgrads == pre-change", float(jnp.max(jnp.abs(flat(g1) - flat(g2)))) == 0.0)
    g3 = G.fu_grads(p, a[0], a[1], Z, Ph, H, "pc")
    rel = float(jnp.max(jnp.abs(flat(g3) - flat(g2))) / jnp.max(jnp.abs(flat(g2))))
    check(f"d{depth} fu_grads('pc') == pc_wgrads", rel < 1e-5, f"{rel:.1e}")
    # at the clean top-down state every hidden forward-update error is zero: only the observation layer gets a gradient
    xtd = G.topdown(p, a[0], Ph, H)[0]
    gf = G.fu_grads(p, a[0], xtd, Z, Ph, H, "fixed")
    hid0 = all(float(jnp.max(jnp.abs(x))) == 0.0 for x in jax.tree_util.tree_leaves((gf["top"], gf["mid"])))
    check(f"d{depth} fixed rule at the top-down state: hidden/top grads exactly 0", hid0)
    # spike 'stable' with c_spike = c is the default schedule
    cs = dc.replace(cfg, spike="stable", c_spike=cfg.c)
    e = G.pc_infer(p, s0, Z, Ph, cs, T=60)
    check(f"d{depth} spike=stable with c_spike=c == default", float(jnp.max(jnp.abs(e[0] - a[0]))) < 1e-6)
    # spike changes only the wavefront steps: with T = j, level j has not spiked yet
    cl = dc.replace(cfg, spike="stable", c=0.1, c_spike=2.0); cn = dc.replace(cfg, c=0.1)
    x1 = G.pc_infer(p, s0, Z, Ph, cl, T=1)[1]; x2 = G.pc_infer(p, s0, Z, Ph, cn, T=1)[1]
    check(f"d{depth} spike at t=0 moves only level 0", float(jnp.max(jnp.abs(x1[0] - x2[0]))) > 0
          and all(float(jnp.max(jnp.abs(u - v))) == 0.0 for u, v in zip(x1[1:], x2[1:])))
    # every arm trains 6 steps (finite or recorded as diverged) and has a distinct name
    zt = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (64, 4, 128)))
    tt = np.asarray(0.5 * jax.random.normal(jax.random.PRNGKey(2), (64, 4, 9)))
    for arm in (dict(spike="lit1"), dict(spike="stable", c=0.1), dict(c=0.1, tag="_c01"), dict(wrule="qi"),
                dict(wrule="fixed"), dict(spike="stable", c=0.1, wrule="fixed")):
        ca = dc.replace(cfg, T=20, T2=10, steps=6, batch=16, **arm)
        _, _, log, div = G.train(ca, zt, tt, verbose=False, log_every=2)
        print(f"     d{depth} {ca.name}: diverged={div} pred_err={[round(r['pred_err'], 2) for r in log]}")
        names.add(ca.name)
        check(f"d{depth} {ca.name}: runs end to end (log written)", len(log) > 0 or div)
    check(f"d{depth} new names differ from stage-1 '{cfg.name}'", cfg.name not in names)
check("all 12 arm names distinct", len(names) == 12, str(sorted(names)))
print("ALL PASS" if OK[0] else "SOME FAIL")
sys.exit(0 if OK[0] else 1)
