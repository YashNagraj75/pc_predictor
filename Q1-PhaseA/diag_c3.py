import dataclasses as dc, glob, itertools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
import pred_gpc as G, pred_ff as PF, pred_eval as PE

# Inference-only follow-up to C2 (never trains).
#  A: on every trained C2 decoder, test k=1 nmse with the belief s from (i) its own rule at T=400, (ii) its own rule at
#     T=2000, (iii) exact auto-decoder inference on the same decoder (c=1, T=2000).  (iii) vs (i) separates belief
#     inference from decoder quality.
#  B: depth-10 multiplier-loop stability: dual rate alpha x primal steps per dual step k x step scale c, on trained
#     depth-10 weights (PC-ALM checkpoints, no-multiplier and BP-trained decoders) and one depth-3 reference.
OUT = sys.argv[1] if len(sys.argv) > 1 else "pred_runs/diag_c3.json"
ztr, ttr, zte, tte, *_ = PF.load("pred_latents.npz", "bp", "emb")
FIELDS = {f.name for f in dc.fields(G.GCfg)}
res = {"A": [], "B": []}
t0 = time.time()


def load(jpath):
    d = json.load(open(jpath))
    cfg = G.GCfg(**{k: v for k, v in d["cfg"].items() if k in FIELDS})
    w = jpath[:-5] + ".eqx"
    if not os.path.exists(w):
        return None, None
    return cfg, eqx.tree_deserialise_leaves(w, G.init(jax.random.PRNGKey(0), cfg))


def ad_predictor(p, cfg, T=2000):
    cb = dc.replace(cfg, rule="bp", c=1.0)
    @eqx.filter_jit
    def _p(p, Zs, Ts, Tt):
        s = G.ad_infer(p, jnp.zeros((Zs.shape[0], G.M)), Zs, G.phi(Ts), cb, T)[0]
        return G.topdown(p, s, G.phi(Tt), cfg.H)[1]
    return lambda Zs, Ts, Tt: _p(p, jnp.asarray(Zs), jnp.asarray(Ts), jnp.asarray(Tt))


# ---------------------------------------------------------------- A
cases1 = PE.k_cases(len(zte), 1)
files = sorted(glob.glob("pred_runs/tests/gpc_*_d*_zero_*_s?.json")) + sorted(glob.glob("pred_runs/tests_amend_c05/*_s?.json"))
for f in files:
    if "w256" in f or "hp9" in f:
        continue
    cfg, p = load(f)
    if p is None:
        continue
    row = {"file": f, "rule": cfg.rule, "depth": cfg.depth, "hid_prec": cfg.hid_prec, "alpha": cfg.alpha, "seed": cfg.seed, "c": cfg.c}
    preds = {"own_T400": G.make_predict(p, None, cfg), "own_T2000": G.make_predict(p, None, cfg, T=2000), "ad_T2000": ad_predictor(p, cfg)}
    for k, fn in preds.items():
        try:
            row[k] = PE.run_cases(fn, zte, tte, *cases1)["nmse"]
        except Exception as e:
            row[k] = None; row[k + "_err"] = str(e)[:120]
    res["A"].append(row)
    print("A", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), f"{time.time() - t0:.0f}s", flush=True)
    G.dump_json(res, OUT)

# ---------------------------------------------------------------- B
rng = np.random.default_rng(7)
idx = rng.choice(len(zte), 512, replace=False)
Zall, Tall = jnp.asarray(zte[idx]), jnp.asarray(tte[idx]); Pall = G.phi(Tall)
perm = np.stack([rng.permutation(Zall.shape[1]) for _ in range(len(idx))])
ii = jnp.arange(len(idx))[:, None]
seen, tgt = jnp.asarray(perm[:, :1]), jnp.asarray(perm[:, 1:])
Zs, Ps, Zt, Pt = Zall[ii, seen], Pall[ii, seen], Zall[ii, tgt], Pall[ii, tgt]
nrm2 = lambda L: sum(jnp.sum(x ** 2) for x in L)


def make_loop(p, cfg):
    # Same update as G.alm_infer; alpha and c are traced, k_inner / T / freeze static (9 compiles per weight set);
    # returns a per-step trace of the plain energy and the relative hidden residual.
    H = cfg.H
    base = G.pc_etas(p, 1, dc.replace(cfg, c=1.0))

    @eqx.filter_jit
    def run(s0, xs0, lam0, Z, Ph, alpha, k_in, c, T_, freeze):
        etas = [c * e for e in base]
        k_s = Z.shape[1]
        etas[0] = c * 0.9 / (cfg.prior + 2.0 * cfg.hid_prec * k_s * G._sn(p["top"][0][:, :G.M]) ** 2)
        Lf = lambda s, xs, lam: G.alm_energy(p, s, xs, Z, Ph, H, cfg.prior, cfg.out_w, cfg.hid_prec, lam)
        gfn = jax.grad(lambda v, lam: Lf(v[0], v[1], lam))

        def step(carry, t):
            (s, xs), lam = carry
            gs, gx = gfn((s, xs), lam)
            s = jnp.where(freeze, s, s - etas[0] * gs)
            xs = [x - e * g for x, g, e in zip(xs, gx, etas[1:])]
            do = ((t + 1) % k_in == 0) & (t < T_ - 1)
            r = G.hid_residuals(p, s, xs, Ph, H)
            lam = [jnp.where(do, l + alpha * ri, l) for l, ri in zip(lam, r)]
            return ((s, xs), lam), (Lf(s, xs, None), jnp.sqrt(nrm2(r) / (nrm2(xs) + 1e-30)))

        return jax.lax.scan(step, ((s0, xs0), lam0), jnp.arange(T_))
    return run


def ad_wg(p, s, cfg):
    return G.ad_wgrads(p, s, Zall, Pall, cfg)


def flat(t):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])


def cosv(a, b):
    a, b = flat(a), flat(b)
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


wsets = sorted(glob.glob("pred_runs/tests/gpc_alm_d10_zero_hp4_a1_k1_bp_s0*.eqx")) + \
    [w for w in ("pred_runs/tests/gpc_alm_d10_zero_hp4_a0_k1_bp_s0.eqx", "pred_runs/tests/gpc_bp_d10_zero_bp_s1.eqx",
                 "pred_runs/tests/gpc_alm_d3_zero_hp4_a1_k1_bp_s0.eqx") if os.path.exists(w)]
GRID = list(itertools.product((1.0, 0.5, 0.25, 0.1), (1, 4, 16), (1.0, 0.5, 0.25)))
for w in wsets:
    depth = 10 if "_d10_" in w else 3
    cfg = G.GCfg(rule="alm", depth=depth, hid_prec=4.0, alpha=1.0, T=400, T2=100)
    p = eqx.tree_deserialise_leaves(w, G.init(jax.random.PRNGKey(0), cfg))
    run = make_loop(p, cfg); H = cfg.H
    s0 = jnp.zeros((len(idx), G.M)); xs0 = G.topdown(p, s0, Ps, H)[0]; lam0 = [jnp.zeros_like(x) for x in xs0]
    s_ad = G.ad_infer(p, s0, Zs, Ps, dc.replace(cfg, rule="bp", c=1.0), 2000)[0]
    for alpha, k_in, c in GRID:
        a_ = jnp.float32(alpha); c_ = jnp.float32(c)
        ((s4, xs4), lam4), (E4, R4) = run(s0, xs0, lam0, Zs, Ps, a_, k_in, c_, 400, False)
        ((s16, xs16), lam16), (E16, R16) = run(s0, xs0, lam0, Zs, Ps, a_, k_in, c_, 1600, False)
        E16, R16, E4 = np.asarray(E16), np.asarray(R16), np.asarray(E4)
        row = {"weights": w, "depth": depth, "alpha": alpha, "k_inner": k_in, "c": c,
               "finite": bool(np.all(np.isfinite(E16))), "E400": float(E4[399]), "E1600": float(E16[-1]),
               "res400": float(np.asarray(R4)[399]), "res1600": float(R16[-1]),
               "osc_last400": float(np.std(E16[-400:]) / (abs(np.mean(E16[-400:])) + 1e-12)),
               "frac_up_2nd_half": float(np.mean(np.diff(E16[800:]) > 0))}
        if row["finite"]:
            row["s_dist_ad_T400"] = float(jnp.mean(jnp.linalg.norm(s4 - s_ad, axis=-1) / (jnp.linalg.norm(s_ad, axis=-1) + 1e-8)))
            row["s_dist_ad_T1600"] = float(jnp.mean(jnp.linalg.norm(s16 - s_ad, axis=-1) / (jnp.linalg.norm(s_ad, axis=-1) + 1e-8)))
            row["nmse_T400"] = float(jnp.mean(jnp.sum((G.topdown(p, s4, Pt, H)[1] - Zt) ** 2, -1)) / jnp.mean(jnp.sum(Zt ** 2, -1)))
            row["nmse_T1600"] = float(jnp.mean(jnp.sum((G.topdown(p, s16, Pt, H)[1] - Zt) ** 2, -1)) / jnp.mean(jnp.sum(Zt ** 2, -1)))
            # phase 2 (all views, s fixed, 100 steps, same schedule) and credit at the T=400 belief
            xa = [x.at[ii, seen].set(xv) for x, xv in zip(G.topdown(p, s4, Pall, H)[0], xs4)]
            la = [jnp.zeros_like(x).at[ii, seen].set(l) for x, l in zip(xa, lam4)]
            ((_, xf), lf), _ = run(s4, xa, la, Zall, Pall, a_, k_in, c_, 100, True)
            ok = all(bool(jnp.all(jnp.isfinite(x))) for x in xf)
            row["credit_cos_T400"] = cosv(G.alm_wgrads(p, s4, xf, lf, Zall, Pall, cfg), ad_wg(p, s4, cfg)) if ok else None
        res["B"].append(row)
        print("B", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), f"{time.time() - t0:.0f}s", flush=True)
        G.dump_json(res, OUT)
G.dump_json(res, OUT)
print("DONE", len(res["A"]), len(res["B"]))
