import dataclasses as dc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, jpc
jax.config.update("jax_default_matmul_precision", "highest")     # same precision as training (review N1)
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
    acts = R.activities(params, x, cfg)
    h = list(acts[:-1])
    # the forward pass equals a float64 NumPy reference of the same network (jpc's scalings and skips)
    prev, err = np.asarray(x, np.float64), 0.0
    for l, m in enumerate(params[0]):
        W, b = np.asarray(m.layers[1].weight, np.float64), np.asarray(m.layers[1].bias, np.float64)
        inp = prev if l == 0 else np.maximum(prev, 0.0)
        nxt = float(a[l]) * (inp @ W.T + b) + (prev if params[1][l] is not None else 0.0)
        err = max(err, float(np.max(np.abs(np.asarray(acts[l], np.float64) - nxt)) / (np.max(np.abs(nxt)) + 1e-12))); prev = nxt
    check(f"d{depth} {arm}: forward pass == float64 reference (rel err < 1e-5)", err < 1e-5, f"{err:.1e}")
    r = R.residuals(params, h, x, a)
    check(f"d{depth} {arm}: residuals zero at the forward pass", max(float(jnp.max(jnp.abs(v))) for v in r) <= 1e-6)
    jz = jpc.init_activities_with_ffwd(params[0], x, skip_model=params[1], param_type=cfg.ptype, gamma=cfg.gamma)
    print(f"     report: |jpc.init_activities_with_ffwd - ours| per layer {[float(jnp.max(jnp.abs(u - v))) for u, v in zip(jz, acts)]}")
    # per-layer activation scale at initialisation (vanishing-signal check, reported)
    print(f"     d{depth} {arm}: rms activation per layer at init {[round(float(jnp.sqrt(jnp.mean(v ** 2))), 4) for v in acts]}")
    print(f"     d{depth} {arm}: scalings {[round(float(v), 4) for v in a]}  skips {[s is not None for s in params[1]]}")
    check(f"d{depth} {arm}: muPC scaling for every rule", cfg.ptype == "mupc" and float(a[0]) < 1.0)
    if arm == "bp":
        continue
    # energy at lam = None equals jpc.pc_energy_fn on a perturbed state
    hp = [v + 0.1 * jax.random.normal(jax.random.PRNGKey(i), v.shape) for i, v in enumerate(h)]
    e_j = jpc.pc_energy_fn(params, hp + [acts[-1]], y, x=x, param_type=cfg.ptype, gamma=cfg.gamma)   # not jitted in jpc
    e_m = R.energy(params, hp, x, y, None, a)
    check(f"d{depth} {arm}: energy == jpc.pc_energy_fn", abs(float(e_j - e_m)) <= 1e-5 * abs(float(e_j)), f"{float(e_j):.6g} vs {float(e_m):.6g}")
    # lam = 0 equals lam = None
    check(f"d{depth} {arm}: zero duals give the plain energy", float(R.energy(params, hp, x, y, [jnp.zeros_like(v) for v in hp], a)) == float(e_m))
cp = R.RCfg(arm="pc", depth=depth, T=50)
params = R.make(jax.random.PRNGKey(depth), 146, 128, cp)
# the activity step is below 2 / lambda_max of the energy Hessian (stability), checked on one sample
x1, y1 = x[:1], y[:1]; a1 = R.scalings(params, x1, cp)
h1 = R.activities(params, x1, cp)[:-1]
fl = jnp.concatenate([v.ravel() for v in h1])
def e_flat(v):
    out, i = [], 0
    for u in h1:
        out.append(v[i:i + u.size].reshape(u.shape)); i += u.size
    return R.energy(params, out, x1, y1, None, a1)
lam_max = float(jnp.max(jnp.linalg.eigvalsh(jax.hessian(e_flat)(fl))))
check(f"d{depth} step * lambda_max(H) < 2", float(R.step_size(params, a1)) * lam_max < 2.0, f"{float(R.step_size(params, a1)) * lam_max:.3f}")
# small output error: the target is the network's own output plus a little noise.  PC-ALM at large T should match
# BP on the same (muPC) network; plain PC is reported for comparison.
y0 = R.forward(params, x, cp) + 1e-3 * jax.random.normal(jax.random.PRNGKey(7), y.shape)
gb = eqx.filter_grad(R.bp_loss)(params, x, y0, cp)
for T in (20, 100, 400):
    c_alm = cos(R.pc_alm_grads(params, x, y0, dc.replace(cp, arm="alm", alpha=1.0, T=T)), gb)
    c_pc = cos(R.pc_alm_grads(params, x, y0, dc.replace(cp, T=T)), gb)
    print(f"     report: small error, T={T}: cos to BP  alm {c_alm:.4f}  plain pc {c_pc:.4f}")
    if T == 400:
        check(f"d{depth} T=400: multipliers move the gradient toward BP (alm >= plain pc)", c_alm >= c_pc, f"{c_alm:.4f} vs {c_pc:.4f}")            # slow modes of a deep chain need more than 400 steps; fail only on a gross mismatch (review N13)
        check(f"d{depth} small error, T=400: PC-ALM gradient close to BP (cos > 0.9)", c_alm > 0.9, f"{c_alm:.4f}")
# power-iteration spectral norm within 3% of the exact value (review N2)
W = params[0][depth // 2].layers[1].weight + 0.05 * jax.random.normal(jax.random.PRNGKey(3), params[0][depth // 2].layers[1].weight.shape)
e = float(R.spec_norm(W) / jnp.linalg.norm(W, 2))
check(f"d{depth} power-iteration spectral norm within [1.0, 1.03] x exact", 1.0 <= e <= 1.03, f"{e:.4f}")
# BP gradient against a central finite difference on one weight entry, float32 tolerance
cb = R.RCfg(arm="bp", depth=depth); pb = R.make(jax.random.PRNGKey(1), 146, 128, cb)
gb_ = eqx.filter_grad(R.bp_loss)(pb, x, y, cb)[0][1].layers[1].weight[3, 5]
bump = lambda d: eqx.tree_at(lambda q: q[0][1].layers[1].weight, pb, pb[0][1].layers[1].weight.at[3, 5].add(d))
fd = (R.bp_loss(bump(1e-1), x, y, cb) - R.bp_loss(bump(-1e-1), x, y, cb)) / 2e-1      # N(0,1) weights: a larger bump
check(f"d{depth} bp gradient == central difference", abs(float(gb_ - fd)) <= 2e-2 * max(abs(float(fd)), 1e-3), f"{float(gb_):.5g} vs {float(fd):.5g}")
# batch independence of the relaxed state
cfa = dc.replace(cp, arm="alm", alpha=1.0, T=20)
_, h1, _ = R.pc_alm_grads(params, x[:8], y[:8], cfa, return_state=True)
_, h2, _ = R.pc_alm_grads(params, x, y, cfa, return_state=True)
d = max(float(jnp.max(jnp.abs(u[:8] - v))) for u, v in zip(h2, h1))
check(f"d{depth} relaxed activities do not depend on the rest of the batch", d < 1e-4, f"{d:.1e}")
print("ALL PASS" if OK[0] else "SOME FAIL")
sys.exit(0 if OK[0] else 1)
