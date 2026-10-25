import dataclasses as dc, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp, equinox as eqx

# Audit of compiled vs uncompiled training gradients at TRAINED weights, for every training path in the project.
# 'eager' = the same function under jax.disable_jit().  A path whose compiled gradient disagrees (cos < 0.999) was
# trained on wrong gradients on this GPU.  ffres BP d10 is a positive control (known bad without the env fix).
#   python audit_jit.py <out.json> [all|sep]
OUT = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
flat = lambda t: jnp.concatenate([v.ravel().astype(jnp.float32) for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])


def cmp(a, b):
    a, b = flat(a), flat(b)
    return {"cos": float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30)), "relmax": float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))}


res = {"env": {"TF32": os.environ.get("NVIDIA_TF32_OVERRIDE"), "XLA_FLAGS": os.environ.get("XLA_FLAGS"), "device": str(jax.devices()[0]), "jax": jax.__version__}, "cases": {}}


# fn takes the arrays as ARGUMENTS so the compiled call traces them (closed-over arrays would be baked in as constants).
def case(name, fn, *args):
    t0 = time.time()
    try:
        gj = eqx.filter_jit(fn)(*args)
        with jax.disable_jit():
            ge = fn(*args)
        res["cases"][name] = {**cmp(gj, ge), "sec": round(time.time() - t0, 1)}
    except Exception as e:
        res["cases"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-800:]}
    print(name, res["cases"][name] if "error" not in res["cases"][name] else res["cases"][name]["error"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


import pred_ff as PF, pred_ffres as RF, pred_gpc as G
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
X, Y, *_ = PF.pairs(ztr, ttr, True)
idx = np.random.default_rng(999).integers(0, len(X), 512); x, y = jnp.asarray(X[idx]), jnp.asarray(Y[idx])
rng = np.random.default_rng(7); bi = rng.choice(len(ztr), 256, replace=False)
Zall, Tall = jnp.asarray(ztr[bi]), jnp.asarray(ttr[bi])
perm = jnp.asarray(np.stack([rng.permutation(4) for _ in range(256)]))


def gcfg(path):
    d = json.load(open(path))["cfg"]
    return G.GCfg(**{f.name: d[f.name] for f in dc.fields(G.GCfg) if f.name in d})


def gpc_grad(p, amort, Zall, Tall, perm, cfg, k=1):
    ii = jnp.arange(Zall.shape[0])[:, None]; seen = perm[:, :k]
    Zs, Ts = Zall[ii, seen], Tall[ii, seen]; Ps, Pall = G.phi(Ts), G.phi(Tall)
    s0 = G.start_s(p, amort, Zs, Ts, cfg)
    if cfg.rule == "sep":
        s = G.sep_belief(p, amort, Zs, Ts, cfg)[0]
        xs2, lam2, _ = G.credit_fixed_s(p, s, Zall, Pall, cfg)
        return G.alm_wgrads(p, s, xs2, lam2, Zall, Pall, cfg), s
    if cfg.rule == "pc":
        s, xs_seen, _, _ = G.pc_infer(p, s0, Zs, Ps, cfg)
    elif cfg.rule == "alm":
        s, xa, la, _, _ = G.alm_infer(p, s0, Zs, Ps, cfg); xs_seen = (xa, la)
    else:
        s = G.ad_infer(p, s0, Zs, Ps, cfg)[0]; xs_seen = None
    return G.weight_grads(p, s, xs_seen, Zall, Pall, seen, cfg)[0], s


def gpc_case(name, base, amort_path=None):
    cfg = gcfg(base + ".json")
    p = eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), cfg, Zall.shape[-1]))
    am = eqx.tree_deserialise_leaves(amort_path, G.init_amort(jax.random.PRNGKey(0), Zall.shape[-1])) if amort_path else None
    case(name, lambda p, am, Z, T, pm: gpc_grad(p, am, Z, T, pm, cfg), p, am, Zall, Tall, perm)


# positive control: residual muPC feedforward BP gradient (known bad on this GPU without the fix)
rc = RF.RCfg(arm="bp", depth=10, seed=0); pb = RF.load_model("pred_runs/frozen_bp/ffres_bp_d10_s0.eqx", rc)
case("CONTROL_ffres_bp_d10", lambda p, a, b: eqx.filter_grad(RF.bp_loss)(p, a, b, rc), pb, x, y)
TS, TT = "pred_runs/tests_sep/", "pred_runs/tests/"
gpc_case("gpc_sep_d10_amort_alm", TS + "gpc_sep_d10_amort_hp16_a1_k1_amort_c1_T2200_bp_s0", TS + "gpc_sep_d10_amort_hp16_a1_k1_amort_c1_T2200_bp_s0_amort.eqx")
gpc_case("gpc_sep_d10_oracle_alm", TS + "gpc_sep_d10_zero_hp16_a1_k1_oracle_c1_T2200_bp_s0")
gpc_case("gpc_sep_d10_pc_none", TS + "gpc_sep_d10_zero_a0_k1_pc_c1_T2200_bp_s0")
gpc_case("gpc_sep_d3_amort_alm", TS + "gpc_sep_d3_amort_hp16_a1_k1_amort_c1_T2200_bp_s0", TS + "gpc_sep_d3_amort_hp16_a1_k1_amort_c1_T2200_bp_s0_amort.eqx")
if MODE == "all":
    gpc_case("gpc_bp_d10_G1", TT + "gpc_bp_d10_zero_bp_s1")
    gpc_case("gpc_bp_d3_G1", TT + "gpc_bp_d3_zero_bp_s1")
    gpc_case("gpc_pc_d10_stage1", TT + "gpc_pc_d10_zero_bp_s1")
    gpc_case("gpc_pc_d3_stage1", TT + "gpc_pc_d3_zero_bp_s1")
    gpc_case("gpc_alm_d10_C2", TT + "gpc_alm_d10_zero_hp4_a1_k1_bp_s1")
    gpc_case("gpc_alm_d3_C2", TT + "gpc_alm_d3_zero_hp4_a1_k1_bp_s1")
    # plain feedforward predictor (pred_ff, sp MLP, 3 layers)
    for nm in ("ff_bp_bp_emb_s0", "ff_pcA_bp_emb_s0", "ff_pcT100_bp_emb_s0", "ff_alm_bp_emb_s0"):
        d = json.load(open(f"pred_runs/{nm}.json")); cd = d.get("cfg", d)
        fc = PF.Cfg(**{f.name: cd[f.name] for f in dc.fields(PF.Cfg) if f.name in cd})
        m = PF.load_model(f"pred_runs/{nm}.eqx", seed=fc.seed)
        fn = {"bp": PF.bp_grads, "alm": PF.alm_grads}.get(fc.arm, PF.pc_grads)
        case(f"ff_{nm}", lambda m, a, b, fc=fc, fn=fn: fn(m, a, b, fc), m, x, y)
    # Phase-A encoder (residual muPC, LeJEPA + SIGReg): backbone BP, plain PC, PC-ALM
    import train as TR, data as D, sigreg as S, run_probes as RP
    for run in ("bp_sigreg_etafix_s0", "pc_sigreg_T16_coupled_etafix_s0", "pc_sigreg_T16_coupled_alm_s0"):
        try:
            ecfg, ep = RP.load_run(f"runs/{run}")
            Xtr, *_ = D.load_dataset(ecfg.data)
            k = jax.random.fold_in(jax.random.PRNGKey(ecfg.opt.seed + 1), 1000); kb, kv, ka, kg = jax.random.split(k, 4)
            ib = jax.random.choice(kb, Xtr.shape[0], (ecfg.opt.batch_size,), replace=False)
            Zin, _ = D.make_views(kv, jnp.asarray(Xtr)[ib], ecfg.data)
            A = S.sample_directions(ka, ecfg.enc.embed_dim, ecfg.sig.num_slices)
            if ecfg.arm == "bp":
                case(f"enc_{run}", lambda p, z, a: TR.bp_grads(p, z, a, ecfg)[0], ep, Zin, A)
            else:
                case(f"enc_{run}", lambda p, z, a, kk: TR.pc_grads(p, z, a, ecfg, kk)[0], ep, Zin, A, kg)
        except Exception as e:
            res["cases"][f"enc_{run}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-800:]}
            print(f"enc_{run}", "ERROR", str(e)[:200], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE")
