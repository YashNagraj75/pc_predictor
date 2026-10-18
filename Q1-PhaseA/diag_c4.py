import dataclasses as dc, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
import pred_gpc as G, pred_ff as PF

# Inference-only test of two fixes from Qi et al. 2025 (arXiv:2506.23800) on trained generative decoders. Never trains.
#  Spiking precision, generative direction: the error wave starts at the observation, so hidden level j gets a one-off
#    large step at t = j and s at t = H.  'lit1' = step 1 on hidden levels (Qi's literal step); 'x10' = 10x the normal
#    curvature-normalised step on hidden levels and s.
#  Forward update: weights learn from eps~ = x_final - mu_td, mu_td = clean top-down pass from the final belief s_T
#    (the computation used at prediction).  'qi' keeps the relaxed presynaptic activity (Qi's rule); 'fixed' uses the
#    top-down one (Millidge's fixed-prediction form).  'pc' is the standard rule and must equal G.pc_wgrads (gate).
OUT = sys.argv[1] if len(sys.argv) > 1 else "pred_runs/diag_c4.json"
FIELDS = {f.name for f in dc.fields(G.GCfg)}
sg = jax.lax.stop_gradient


def flat(t):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])


def cosv(a, b):
    a, b = flat(a), flat(b)
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


def blocks(g):
    return [g["top"]] + list(g["mid"]) + [g["bot"]]


# Relaxation with a per-level spike at the wavefront; mode 'none' is G.pc_infer exactly (gated).
def infer_sched(p, cfg, s0, Z, Ph, T, mode, freeze_s=False, xs0=None):
    H = cfg.H
    xs0 = G.topdown(p, s0, Ph, H)[0] if xs0 is None else xs0
    etas = G.pc_etas(p, Z.shape[1], cfg)
    Ef = G._E(cfg)
    gfn = jax.grad(lambda v: Ef(p, v[0], v[1], Z, Ph))
    if mode == "lit1":
        mx, ms = [1.0 / e for e in etas[1:]], 1.0
    elif mode == "x10":
        mx, ms = [10.0] * H, 10.0
    else:
        mx, ms = [1.0] * H, 1.0

    def step(v, t):
        gs, gx = gfn(v)
        s = v[0] if freeze_s else v[0] - etas[0] * jnp.where(t == H, ms, 1.0) * gs
        xs = [x - etas[j + 1] * jnp.where(t == j, mx[j], 1.0) * g for j, (x, g) in enumerate(zip(v[1], gx))]
        return (s, xs), None

    (s, xs), _ = jax.lax.scan(step, (s0, xs0), jnp.arange(T))
    return s, xs


# Weight gradient from fixed errors and presynaptic activities via a linear surrogate (dE/dW = -eps * dmu/dW).
def fu_grads(p, s, xs, Z, Ph, H, mode):
    s, xs = sg(s), [sg(x) for x in xs]
    xtd = [sg(x) for x in G.topdown(p, s, Ph, H)[0]]
    if mode == "pc":
        pre = xs
        e0 = Z - G.mu_bot(p, xs[0])
        em = [xs[l - 1] - G.mu_mid(p["mid"][l - 1], xs[l], H) for l in range(1, H)]
    else:
        pre = xs if mode == "qi" else xtd
        e0 = Z - G.mu_bot(p, xtd[0])
        em = [xs[l - 1] - xtd[l - 1] for l in range(1, H)]
    eH = xs[H - 1] - G.mu_top(p, s, Ph)
    e0, em, eH = sg(e0), [sg(e) for e in em], sg(eH)

    def sur(q):
        v = jnp.sum(e0 * G.mu_bot(q, pre[0])) + jnp.sum(eH * G.mu_top(q, s, Ph))
        v += sum(jnp.sum(e * G.mu_mid(q["mid"][l - 1], pre[l], H)) for l, e in zip(range(1, H), em))
        return -v
    g = jax.grad(sur)(p)
    return jax.tree_util.tree_map(lambda x: x / Z.shape[0], g)


def shares(e):
    e = np.asarray([float(v) for v in e]); e = e / e.sum()
    return {"obs": float(e[0]), "top": float(e[-1]), "min": float(e.min()),
            "log10_max_over_min": float(np.log10(e.max() / max(e.min(), 1e-30))), "all": e.round(5).tolist()}


ztr, ttr, zte, tte, *_ = PF.load("pred_latents.npz", "bp", "emb")
rng = np.random.default_rng(7)
idx = rng.choice(len(zte), 512, replace=False)
Zall, Tall = jnp.asarray(zte[idx]), jnp.asarray(tte[idx]); Pall = G.phi(Tall)
perm = np.stack([rng.permutation(Zall.shape[1]) for _ in range(len(idx))])
ii = jnp.arange(len(idx))[:, None]
seen, tgt = jnp.asarray(perm[:, :1]), jnp.asarray(perm[:, 1:])
Zs, Ps, Zt, Pt = Zall[ii, seen], Pall[ii, seen], Zall[ii, tgt], Pall[ii, tgt]
sel = json.load(open("pred_runs/gpc_selected.json"))
res = {"gates": [], "rows": []}
t0 = time.time()

WSETS = [("d10", "pc_s1", "pred_runs/tests/gpc_pc_d10_zero_bp_s1"), ("d10", "pc_s2", "pred_runs/tests/gpc_pc_d10_zero_bp_s2"),
         ("d10", "bp_s1", "pred_runs/tests/gpc_bp_d10_zero_bp_s1"), ("d10", "nomult_s0", "pred_runs/tests/gpc_alm_d10_zero_hp4_a0_k1_bp_s0"),
         ("d10", "alm_s0_step1000", "pred_runs/tests/gpc_alm_d10_zero_hp4_a1_k1_bp_s0_step1000"),
         ("d3", "pc_s1", "pred_runs/tests/gpc_pc_d3_zero_bp_s1"), ("d3", "bp_s1", "pred_runs/tests/gpc_bp_d3_zero_bp_s1"),
         ("d3", "alm_s0", "pred_runs/tests/gpc_alm_d3_zero_hp4_a1_k1_bp_s0")]

for dtag, name, base in WSETS:
    if not os.path.exists(base + ".eqx"):
        print("MISSING", base, flush=True); res["rows"].append({"weights": name, "depth": dtag, "missing": True}); continue
    depth = int(dtag[1:])
    c_pc = sel.get(f"gpc_pc_d{depth}_zero", {}).get("c", 2.0)
    cfg = G.GCfg(rule="pc", depth=depth, hid_prec=1.0, c=c_pc, T=400, T2=100)
    p = eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), cfg)); H = cfg.H
    s0 = jnp.zeros((len(idx), G.M))
    s_ad = G.ad_infer(p, s0, Zs, Ps, dc.replace(cfg, rule="bp", c=1.0), 400)[0]
    g_ad_sad = G.ad_wgrads(p, s_ad, Zall, Pall, cfg)
    nm = lambda s: float(jnp.mean(jnp.sum((G.topdown(p, s, Pt, H)[1] - Zt) ** 2, -1)) / jnp.mean(jnp.sum(Zt ** 2, -1)))

    if name in ("pc_s1",):          # gates once per depth
        sa, xa = jax.jit(lambda s0: infer_sched(p, cfg, s0, Zs, Ps, 400, "none"))(s0)
        sb, xb, _, _ = G.pc_infer(p, s0, Zs, Ps, cfg, T=400)
        d_inf = max(float(jnp.max(jnp.abs(sa - sb))), max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(xa, xb)))
        gp = G.pc_wgrads(p, sb, xb, Zs, Ps, cfg); gf = fu_grads(p, sb, xb, Zs, Ps, H, "pc")
        d_g = float(jnp.max(jnp.abs(flat(gp) - flat(gf))) / (jnp.max(jnp.abs(flat(gp))) + 1e-30))
        xtd = G.topdown(p, s_ad, Ps, H)[0]
        c_fix_td = cosv(fu_grads(p, s_ad, xtd, Zs, Ps, H, "fixed"), G.ad_wgrads(p, s_ad, Zs, Ps, cfg))
        gate = {"depth": depth, "sched_none_eq_pc_infer_maxabs": d_inf, "fu_pc_eq_pc_wgrads_relmax": d_g,
                "fixed_at_topdown_state_cos_ad_bottom_only": c_fix_td,
                "pass": bool(d_inf < 1e-4 and d_g < 1e-4)}
        res["gates"].append(gate); print("GATE", json.dumps(gate), flush=True); G.dump_json(res, OUT)

    for T in sorted({H + 1, 2 * H + 2, 50, 400}):
        T2 = min(T, 100)
        for mode in ("none", "lit1", "x10"):
            row = {"weights": name, "depth": depth, "T": T, "T2": T2, "spike": mode, "c": c_pc}
            s, xs = jax.jit(lambda s0: infer_sched(p, cfg, s0, Zs, Ps, T, mode))(s0)
            fin = bool(jnp.all(jnp.isfinite(s))) and all(bool(jnp.all(jnp.isfinite(x))) for x in xs)
            row["finite"] = fin
            if not fin:
                res["rows"].append(row); print("ROW", json.dumps(row), flush=True); continue
            row["share_phase1"] = shares(G.level_energies(p, s, xs, Zs, Ps, H))
            row["nmse_heldout_k1"] = nm(s)
            row["s_dist_ad"] = float(jnp.mean(jnp.linalg.norm(s - s_ad, axis=-1) / (jnp.linalg.norm(s_ad, axis=-1) + 1e-8)))
            # phase 2: all views, s fixed, seen views warm-started, same spike schedule
            x0 = [x.at[ii, seen].set(xv) for x, xv in zip(G.topdown(p, s, Pall, H)[0], xs)]
            _, x2 = jax.jit(lambda s, x0: infer_sched(p, cfg, s, Zall, Pall, T2, mode, freeze_s=True, xs0=x0))(s, x0)
            row["share_phase2"] = shares(G.level_energies(p, s, x2, Zall, Pall, H))
            g_ad = G.ad_wgrads(p, s, Zall, Pall, cfg)
            for um in ("pc", "qi", "fixed"):
                g = fu_grads(p, s, x2, Zall, Pall, H, um)
                if not bool(jnp.all(jnp.isfinite(flat(g)))):
                    row[f"cos_{um}"] = None; continue
                row[f"cos_{um}"] = cosv(g, g_ad)
                row[f"cos_{um}_blocks"] = [round(cosv(a, b), 4) for a, b in zip(blocks(g), blocks(g_ad))]
                row[f"ratio_{um}"] = float(jnp.linalg.norm(flat(g)) / (jnp.linalg.norm(flat(g_ad)) + 1e-30))
                row[f"cos_{um}_vs_ad_at_s_ad"] = cosv(g, g_ad_sad)
            res["rows"].append(row)
            print("ROW", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if "blocks" not in k and "share" not in k}),
                  f"{time.time() - t0:.0f}s", flush=True)
            G.dump_json(res, OUT)
G.dump_json(res, OUT)
print("DONE", len(res["rows"]))
