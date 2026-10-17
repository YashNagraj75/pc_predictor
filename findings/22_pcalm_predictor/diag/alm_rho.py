import sys, json
sys.path.insert(0, "/home/yash/pc_predictor/Q1-PhaseA")
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, jpc, pred_ff as PF
D = sys.argv[1]
ztr, ttr, *_ = PF.load("/home/yash/pc_predictor/Q1-PhaseA/pred_latents.npz", "bp", "emb")
X, Y, *_ = PF.pairs(ztr[:4000], ttr[:4000], True)
idx = np.random.default_rng(123).integers(0, len(X), 512); x, y = jnp.asarray(X[idx]), jnp.asarray(Y[idx])
L = lambda g: jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))])
cos = lambda a, b: float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
def energy(m, acts, lam, rho):
    preds = PF._layer_preds(m, acts, x)
    r = [a - f for a, f in zip(acts[:-1], preds[:-1])]
    return 0.5 * (rho * sum(jnp.sum((ri + li) ** 2) for ri, li in zip(r, lam)) + jnp.sum((y - preds[-1]) ** 2)) / x.shape[0]
def run(m, rho, T, k):
    acts = list(jpc.init_activities_with_ffwd(m, x, param_type="sp")); B = x.shape[0]; yl = acts[-1]
    sq = [jnp.linalg.norm(l.layers[1].weight if hasattr(l, "layers") else l.weight, 2) ** 2 for l in m[1:]]
    eta = 0.9 / (rho + 2.0 * max(rho, 1.0) * jnp.max(jnp.stack(sq)))
    lam = [jnp.zeros_like(a) for a in acts[:-1]]; h = acts[:-1]
    gfn = jax.jit(jax.grad(lambda h, lam: energy(m, h + [yl], lam, rho)))
    for t in range(T):
        g = gfn(h, lam); h = [a - eta * B * ga for a, ga in zip(h, g)]
        if (t + 1) % k == 0 and t < T - 1:
            preds = PF._layer_preds(m, h + [yl], x); lam = [li + (a - f) for li, a, f in zip(lam, h, preds[:-1])]
    preds = PF._layer_preds(m, h + [yl], x)
    res = float(jnp.sqrt(sum(jnp.sum((a - f) ** 2) for a, f in zip(h, preds[:-1])) / sum(jnp.sum(a ** 2) for a in h)))
    g = eqx.filter_grad(lambda mm: energy(mm, h + [yl], lam, rho))(m)
    return g, res
models = {"init": PF.make(jax.random.PRNGKey(0), 146, 128)}
for s in (0, 2):
    models[f"alm_s{s}"] = PF.load_model(f"{D}/ff_alm_bp_emb_s{s}.eqx", seed=s)
models["bp_s0"] = PF.load_model(f"{D}/ff_bp_bp_emb_s0.eqx", seed=0)
for name, m in models.items():
    gb = L(PF.bp_grads(m, x, y, None)); row = []
    for rho in (1.0, 4.0, 16.0, 64.0):
        for T, k in ((100, 1), (400, 1), (400, 10)):
            g, res = run(m, rho, T, k)
            row.append(f"rho{rho:g} T{T} k{k}: {cos(L(g), gb):.3f}/{res:.0e}")
    print(name, " | ".join(row), flush=True)
