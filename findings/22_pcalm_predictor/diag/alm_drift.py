import sys, json, dataclasses
sys.path.insert(0, "/home/yash/pc_predictor/Q1-PhaseA")
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, jpc, pred_ff as PF
D = sys.argv[1]
ztr, ttr, *_ = PF.load("/home/yash/pc_predictor/Q1-PhaseA/pred_latents.npz", "bp", "emb")
X, Y, *_ = PF.pairs(ztr[:4000], ttr[:4000], True)
idx = np.random.default_rng(123).integers(0, len(X), 512); x, y = jnp.asarray(X[idx]), jnp.asarray(Y[idx])
L = lambda g: jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))])
cos = lambda a, b: float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
def lcos(ga, gb): return [round(cos(L(ga[i]), L(gb[i])), 3) for i in range(3)]

def alm_k(m, cfg, k):
    # PC-ALM with k primal steps per dual step (k=1 is the training rule)
    acts = list(jpc.init_activities_with_ffwd(m, x, param_type="sp")); B = x.shape[0]
    sq = [jnp.linalg.norm(l.layers[1].weight if hasattr(l, "layers") else l.weight, 2) ** 2 for l in m[1:]]
    eta = 0.9 / (1.0 + 2.0 * jnp.max(jnp.stack(sq)))
    lam = [jnp.zeros_like(a) for a in acts[:-1]]; h = acts[:-1]
    gfn = jax.jit(jax.grad(lambda h, lam: PF.alm_energy(m, h + [acts[-1]], x, y, lam)))
    hist = []
    for t in range(cfg.T):
        for _ in range(k):
            g = gfn(h, lam); h = [a - eta * B * ga for a, ga in zip(h, g)]
        if t < cfg.T - 1:
            preds = PF._layer_preds(m, h + [acts[-1]], x)
            new = [li + cfg.alpha * (a - f) for li, a, f in zip(lam, h, preds[:-1])]
            hist.append(float(sum(jnp.sum((n - o) ** 2) for n, o in zip(new, lam)) ** 0.5)); lam = new
    preds = PF._layer_preds(m, h + [acts[-1]], x)
    res = float(sum(jnp.sum((a - f) ** 2) for a, f in zip(h, preds[:-1])) ** 0.5 / jnp.sqrt(B))
    g = eqx.filter_grad(lambda mm: PF.alm_energy(mm, h + [acts[-1]], x, y, lam))(m)
    return g, res, hist, float(eta)

out = {}
models = {"init": PF.make(jax.random.PRNGKey(0), 146, 128)}
for s in (0, 1, 2):
    models[f"alm_s{s}"] = PF.load_model(f"{D}/ff_alm_bp_emb_s{s}.eqx", seed=s)
    models[f"bp_s{s}"] = PF.load_model(f"{D}/ff_bp_bp_emb_s{s}.eqx", seed=s)
for name, m in models.items():
    gb = PF.bp_grads(m, x, y, None)
    sq = [float(jnp.linalg.norm(l.layers[1].weight if hasattr(l, "layers") else l.weight, 2) ** 2) for l in m[1:]]
    out_err = float(jnp.mean(jnp.sum((y - PF.forward(m, x)) ** 2, -1)))
    row = {"W2_layers2_3": [round(v, 2) for v in sq], "out_err": round(out_err, 2), "pc_T100": round(cos(L(PF.pc_grads(m, x, y, PF.Cfg(T=100, adaptive_eta=True))), L(gb)), 3)}
    for T in (20, 100, 400, 2000):
        g, res, hist, eta = alm_k(m, PF.Cfg(T=T, alpha=1.0), 1)
        row[f"alm_T{T}"] = (round(cos(L(g), L(gb)), 4), lcos(g, gb), round(float(jnp.linalg.norm(L(g)) / jnp.linalg.norm(L(gb))), 3), round(res, 4), round(hist[-1], 4))
    row["eta"] = round(eta, 4)
    for k in (5, 20):
        g, res, hist, _ = alm_k(m, PF.Cfg(T=100, alpha=1.0), k)
        row[f"alm_T100_k{k}"] = (round(cos(L(g), L(gb)), 4), lcos(g, gb))
    out[name] = row; print(name, json.dumps(row), flush=True)
json.dump(out, open(f"{D}/alm_drift.json", "w"), indent=1)
