import dataclasses as dc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, jpc
import pred_ffres as R

# Gates for pred_ffres.py; run one depth per process: python test_ffres.py <depth>.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def flat(t):
    return jnp.concatenate([v.ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])


def cos(a, b):
    a, b = flat(a), flat(b)
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


depth = int(sys.argv[1]) if len(sys.argv) > 1 else 10
kx, ky = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(kx, (64, 146)); y = jax.random.normal(ky, (64, 128))
for arm in ("bp", "pc", "alm"):
    cfg = R.RCfg(arm=arm, depth=depth)
    params = R.make(jax.random.PRNGKey(depth), 146, 128, cfg)
    a = R.scalings(params, x, cfg)
    acts = jpc.init_activities_with_ffwd(params[0], x, skip_model=params[1], param_type=cfg.ptype, gamma=cfg.gamma)
    h = list(acts[:-1])
    # the hand-written layer predictions reproduce jpc's forward pass (all residuals zero at the forward pass)
    r = R.residuals(params, h, x, a)
    out = R.layer_pred(params, depth - 1, h[-1], a)
    check(f"d{depth} {arm} ({cfg.ptype}): residuals zero at jpc's forward pass", max(float(jnp.max(jnp.abs(v))) for v in r) < 1e-5
          and float(jnp.max(jnp.abs(out - acts[-1]))) < 1e-5)
    # per-layer activation scale at initialisation (vanishing-signal check, reported)
    print(f"     d{depth} {arm}: rms activation per layer at init {[round(float(jnp.sqrt(jnp.mean(v ** 2))), 4) for v in acts]}")
    print(f"     d{depth} {arm}: scalings {[round(float(v), 4) for v in a]}  skips {[s is not None for s in params[1]]}")
    if arm == "bp":
        check(f"d{depth} bp: standard parameterisation, no muPC scaling", all(float(v) == 1.0 for v in a))
        continue
    check(f"d{depth} {arm}: muPC scaling", cfg.ptype == "mupc" and float(a[0]) < 1.0)
    # energy at lam = None equals jpc.pc_energy_fn on a perturbed state
    hp = [v + 0.1 * jax.random.normal(jax.random.PRNGKey(i), v.shape) for i, v in enumerate(h)]
    e_j = jpc.pc_energy_fn(params, hp + [acts[-1]], y, x=x, param_type=cfg.ptype, gamma=cfg.gamma)
    e_m = R.energy(params, hp, x, y, None, a)
    check(f"d{depth} {arm}: energy == jpc.pc_energy_fn", abs(float(e_j - e_m)) <= 1e-5 * abs(float(e_j)), f"{float(e_j):.6g} vs {float(e_m):.6g}")
    # lam = 0 equals lam = None
    check(f"d{depth} {arm}: zero duals give the plain energy", float(R.energy(params, hp, x, y, [jnp.zeros_like(v) for v in hp], a)) == float(e_m))
# alpha = 0 (pc arm) gives the same gradient as alm with alpha = 0
cp = R.RCfg(arm="pc", depth=depth, T=50); ca = dc.replace(cp, arm="alm", alpha=0.0)
params = R.make(jax.random.PRNGKey(depth), 146, 128, cp)
d = float(jnp.max(jnp.abs(flat(R.pc_alm_grads(params, x, y, cp)) - flat(R.pc_alm_grads(params, x, y, ca)))))
check(f"d{depth} pc arm == alm with alpha = 0", d < 1e-7, f"{d:.1e}")
# small output error: the target is the network's own output plus a little noise.  PC-ALM at large T should match
# BP on the same (muPC) network; plain PC is reported for comparison.
y0 = R.forward(params, x, cp) + 1e-3 * jax.random.normal(jax.random.PRNGKey(7), y.shape)
gb = eqx.filter_grad(R.bp_loss)(params, x, y0, cp)
for T in (20, 100, 400):
    c_alm = cos(R.pc_alm_grads(params, x, y0, dc.replace(cp, arm="alm", alpha=1.0, T=T)), gb)
    c_pc = cos(R.pc_alm_grads(params, x, y0, dc.replace(cp, T=T)), gb)
    print(f"     report: small error, T={T}: cos to BP  alm {c_alm:.4f}  plain pc {c_pc:.4f}")
    if T == 400:
        check(f"d{depth} small error, T=400: PC-ALM gradient matches BP (cos > 0.99)", c_alm > 0.99, f"{c_alm:.4f}")
# batch independence of the relaxed state
cfa = dc.replace(cp, arm="alm", alpha=1.0, T=20)
_, h1, _ = R.pc_alm_grads(params, x[:8], y[:8], cfa, return_state=True)
_, h2, _ = R.pc_alm_grads(params, x, y, cfa, return_state=True)
d = max(float(jnp.max(jnp.abs(u[:8] - v))) for u, v in zip(h2, h1))
check(f"d{depth} relaxed activities do not depend on the rest of the batch", d < 1e-4, f"{d:.1e}")
print("ALL PASS" if OK[0] else "SOME FAIL")
sys.exit(0 if OK[0] else 1)
