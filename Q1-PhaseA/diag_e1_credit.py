import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import pred_ff as PF, pred_ffres as R

# Inference-only check on trained E1 checkpoints: does PC-ALM's weight gradient return to BP's (same network, same
# batch) with longer inference or a larger penalty?  Recovery = convergence budget; staying low = a loop problem.
# Scaled method of multipliers: E = (rho sum ||r_l + u_l||^2 + ||y - yhat||^2) / 2B, u_l += alpha r_l after every step
# but the last.  rho = 1 reproduces pred_ffres.pc_alm_grads exactly (gate); at the saddle rho u_l = BP's error delta_l.
OUT = sys.argv[1] if len(sys.argv) > 1 else "pred_runs/diag_e1_credit.json"
TS = (100, 400, 2000, 10000)
GRID = [("trained", None, None), ("rho1_a1", 1.0, 1.0), ("rho4_a1", 4.0, 1.0), ("rho16_a1", 16.0, 1.0), ("plainpc", 1.0, 0.0)]
CKPTS = [("alm", 10, 0), ("alm", 10, 1), ("alm", 10, 2), ("pc", 10, 0), ("alm", 3, 0)]


def flat(t):
    return jnp.concatenate([v.ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])


def cos(u, v):
    u, v = flat(u), flat(v)
    return float(u @ v / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-30))


def energy(params, h, x, y, u, a, rho):
    r = R.residuals(params, h, x, a)
    out = y - R.layer_pred(params, len(h), h[-1], a)
    return 0.5 * (rho * sum(jnp.sum((ri + ui) ** 2) for ri, ui in zip(r, u)) + jnp.sum(out ** 2)) / x.shape[0]


def run(params, x, y, cfg, T, rho, alpha):
    a = R.scalings(params, x, cfg)
    h0 = R.activities(params, x, cfg)[:-1]
    u0 = [jnp.zeros_like(v) for v in h0]
    eta, B = R.step_size(params, a) / max(1.0, rho), x.shape[0]
    gfn = jax.grad(lambda h, u: energy(params, h, x, y, u, a, rho))

    def step(c, t):
        h, u = c
        h = [v - eta * B * g for v, g in zip(h, gfn(h, u))]
        r = R.residuals(params, h, x, a)
        u = [jnp.where(t < T - 1, ui + alpha * ri, ui) for ui, ri in zip(u, r)]
        return (h, u), None

    (h, u), _ = jax.lax.scan(step, (h0, u0), jnp.arange(T))
    h, u = jax.lax.stop_gradient(h), jax.lax.stop_gradient(u)
    g = eqx.filter_grad(lambda p: energy(p, h, x, y, u, a, rho))(params)
    r = R.residuals(params, h, x, a)
    out = y - R.layer_pred(params, len(h), h[-1], a)
    return g, h, u, r, out


# Run eagerly (the scan body is still compiled): under an outer eqx.filter_jit on the host GPU this function gave a
# different state from the identical eager code (activities moved ~12x further; dbg 2026-10-25).
def make_run(cfg):
    return lambda params, x, y, T, rho, alpha: run(params, x, y, cfg, T, rho, alpha)


# Does an outer filter_jit change the TRAINING gradient?  Same function, same inputs, compiled vs eager.
def jit_check(params, x, y, cfg):
    ge = R.grads(params, x, y, cfg)
    gj = eqx.filter_jit(lambda p, a, b: R.grads(p, a, b, cfg))(params, x, y)
    return {"rel_maxdiff": float(jnp.max(jnp.abs(flat(ge) - flat(gj))) / (jnp.max(jnp.abs(flat(ge))) + 1e-30)), "cos": cos(ge, gj)}


# BP's error at each hidden activity: delta_l = dL_BP/dh_l, via additive perturbations of the forward pass.
def bp_deltas(params, x, y, cfg):
    a = R.scalings(params, x, cfg)
    def loss(eps):
        prev = x
        for l in range(len(params[0])):
            prev = R.layer_pred(params, l, prev, a) + (eps[l] if l < len(eps) else 0.0)
        return 0.5 * jnp.sum((y - prev) ** 2) / x.shape[0]
    h0 = R.activities(params, x, cfg)[:-1]
    return jax.grad(loss)([jnp.zeros_like(v) for v in h0])


def main():
    ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
    X, Y, *_ = PF.pairs(ztr, ttr, True)
    pidx = np.random.default_rng(999).integers(0, len(X), 512)       # the training run's fixed probe batch
    x, y = jnp.asarray(X[pidx]), jnp.asarray(Y[pidx])
    res = {"gates": [], "rows": []}
    for arm, depth in (("bp", 3), ("bp", 10)):             # BP training gradient, compiled vs eager
        cfg = R.RCfg(arm=arm, depth=depth, seed=0)
        params = R.load_model(f"pred_runs/frozen_bp/ffres_bp_d{depth}_s0.eqx", cfg)
        res["gates"].append({"ckpt": f"bp_d{depth}_s0", "train_grad_jit_vs_eager": jit_check(params, x, y, cfg)}); print("GATE", res["gates"][-1], flush=True)
    for arm, depth, seed in CKPTS:
        src = json.load(open(f"pred_runs/ffres/ffres_{arm}_d{depth}_s{seed}.json"))
        cfg = R.RCfg(arm=arm, depth=depth, seed=seed, T=src["cfg"]["T"], alpha=src["cfg"]["alpha"], lr=src["cfg"]["lr"])
        params = R.load_model(f"pred_runs/ffres/ffres_{arm}_d{depth}_s{seed}.eqx", cfg)
        gb = eqx.filter_grad(R.bp_loss)(params, x, y, cfg)
        dl = bp_deltas(params, x, y, cfg)
        e_bp = float(jnp.linalg.norm(y - R.forward(params, x, cfg)))
        # gate: rho 1 at the trained (T, alpha) reproduces pred_ffres and the last logged credit cosine
        g_ref = R.pc_alm_grads(params, x, y, cfg)
        run_j = make_run(cfg)
        g_me = run_j(params, x, y, cfg.T, 1.0, cfg.alpha if arm == "alm" else 0.0)[0]
        gd = float(jnp.max(jnp.abs(flat(g_ref) - flat(g_me))) / (jnp.max(jnp.abs(flat(g_ref))) + 1e-30))
        logged = [r_ for r_ in src["log"] if "cos_bp" in r_][-1]["cos_bp"]
        res["gates"].append({"ckpt": f"{arm}_d{depth}_s{seed}", "rel_diff_vs_pred_ffres": gd, "cos_now": cos(g_ref, gb), "cos_logged_last": logged,
                             "train_grad_jit_vs_eager": jit_check(params, x, y, cfg)})
        print("GATE", res["gates"][-1], flush=True)
        for name, rho, alpha in GRID:
            if name == "trained":
                rho, alpha = 1.0, (cfg.alpha if arm == "alm" else 0.0)
            for T in TS:
                t0 = time.time()
                g, h, u, r, out = run_j(params, x, y, T, rho, alpha)
                fin = bool(jnp.all(jnp.isfinite(flat(g))))
                row = {"ckpt": f"{arm}_d{depth}_s{seed}", "arm": arm, "depth": depth, "seed": seed, "setting": name, "rho": rho,
                       "alpha": alpha, "T": T, "finite": fin}
                if fin:
                    row.update({"cos": cos(g, gb), "ratio": float(jnp.linalg.norm(flat(g)) / jnp.linalg.norm(flat(gb))),
                                "layer_cos": [round(cos(p, q), 4) for p, q in zip(g[0], gb[0])],
                                "lam_cos_delta": [round(float(jnp.sum(rho * ui * di) / (jnp.linalg.norm(rho * ui) * jnp.linalg.norm(di) + 1e-30)), 4) for ui, di in zip(u, dl)],
                                "lam_ratio_delta": [round(float(jnp.linalg.norm(rho * ui) / (jnp.linalg.norm(di) + 1e-30)), 4) for ui, di in zip(u, dl)],
                                "res_rel": float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in r) / sum(jnp.sum(v ** 2) for v in h))),
                                "out_ratio": float(jnp.linalg.norm(out)) / e_bp})
                row["sec"] = round(time.time() - t0, 1)
                res["rows"].append(row)
                print("ROW", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("layer_cos", "lam_cos_delta", "lam_ratio_delta")}),
                      "layer_cos", row.get("layer_cos"), flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    print("DONE", len(res["rows"]))


if __name__ == "__main__":
    main()
