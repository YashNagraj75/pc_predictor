"""Why does the feedforward PC predictor lose to BP and blow up late?  Dense log of one PC run (bp backbone, seed 0)."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax, pred_ff as P
cfg = P.Cfg(backbone="bp", layer="emb", arm="pc", seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0, steps=6000, T=20, eta_h=0.05, adaptive_eta=len(sys.argv) > 2 and sys.argv[2] == "adaptive")
ztr, ttr, zte, tte, *_ = P.load("pred_latents.npz", "bp", "emb")
X, Y, *_ = P.pairs(ztr, ttr)
model = P.make(jax.random.PRNGKey(cfg.seed), X.shape[1], Y.shape[1])
opt = P.optimiser(cfg); st = opt.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def step(model, st, x, y):
    g, acts, F = P.pc_grads(model, x, y, cfg, return_acts=True)
    upd, st = opt.update(g, st, eqx.filter(model, eqx.is_inexact_array))
    yff = P.forward(model, x)
    e_ff = jnp.linalg.norm(y - yff); e_eq = jnp.linalg.norm(y - jax.vmap(model[-1])(acts[-2]))
    lip = [jnp.linalg.norm(l.layers[1].weight if hasattr(l, "layers") else l.weight, 2) ** 2 for l in model]
    return eqx.apply_updates(model, upd), st, F, e_eq / e_ff, lip, optax.global_norm(eqx.filter(g, eqx.is_inexact_array))

rng = np.random.default_rng(cfg.seed + 1000); rows = []
for s in range(cfg.steps):
    idx = rng.integers(0, len(X), cfg.batch)
    model, st, F, absorb, lip, gn = step(model, st, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))
    if s % 250 == 0 or s == cfg.steps - 1:
        F = np.asarray(F)
        r = {"step": s, "F_first": float(F[0]), "F_last": float(F[-1]), "F_max_increase": float(np.max(np.diff(F))),
             "monotone": bool(np.all(np.diff(F) <= 1e-4 * abs(F[0]) + 1e-8)), "out_err_eq_over_ff": float(absorb),
             "W_spec_sq": [round(float(v), 2) for v in lip], "gnorm": float(gn)}
        if s % 1000 == 0 or s == cfg.steps - 1:
            r["test"] = P.evaluate(model, zte[:500], tte[:500], n_retr=500)
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(rows, open(f"pred_runs/diag_pc_s{cfg.seed}{'_adaptive' if cfg.adaptive_eta else ''}.json", "w"), indent=1)
