import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
import data as D, vit_ijepa as V, train_pred_vit as A
# Per-step training cost of BP, gain-matched ePC and standard PC at the final stage-A settings (rev 2 after 3 reviewers):
# each arm's trained predictor, one fixed real batch, the exact training calls (encode + step, gain amortised at its
# every-50-steps schedule), compiled first, then ROUNDS rotated rounds of N steps; GPU state sampled around every round.
# python timing_vit.py <out.json> [N per round] [rounds]   (inside the GPU lock, with the job_stageA.sh environment)
OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10; ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
assert jax.devices()[0].platform == "gpu", jax.devices()
assert "--xla_gpu_enable_triton_gemm=false" in os.environ.get("XLA_FLAGS", ""), "training ran with Triton GEMMs off"
assert jax.config.jax_default_matmul_precision == "highest" and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
SA = "runs_vit/stageA"; RUNS = {"bp": "bp_d{d}_lr3e-3_e25_full_s0", "epc_alm_blk4": "epc_alm_blk4_d{d}_lr3e-3_e25_full_s0_T20g12",
                               "spc_u1": "spc_u1_d{d}_lr3e-3_e25_full_s0_T20g12"}
Q = "utilization.gpu,memory.used,clocks.sm,clocks.max.sm,power.draw,temperature.gpu"


def gpu_state():
    try:
        g = subprocess.run(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        return {"gpu": dict(zip(Q.split(","), [x.strip() for x in g.split(",")])), "n_compute_apps": len([l for l in a.splitlines() if l.strip()]), "apps": a}
    except Exception as e:
        return {"error": str(e)}


def sync(t):
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, t); return t


vc, enc, tenc, _ = A.load_encoders()
Xtr, _, _, _ = D.load_cifar10(n_train=-1, n_test=2000, seed=0)
res = {"env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "NVIDIA_TF32_OVERRIDE")}, "device": str(jax.devices()[0]), "N_per_round": N,
       "rounds": ROUNDS, "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "idle_before": gpu_state(), "rows": []}
for depth in (10, 6):
    rng = np.random.default_rng([7, depth]); a0 = A.ACfg(depth=depth)
    imgs = jnp.asarray(V.standardise(V.augment(rng, Xtr[:a0.batch])))
    m = tuple(jnp.asarray(x) for x in V.sample_masks(rng, a0.batch, vc))
    enc_call = lambda: A.encode(enc, tenc, imgs, imgs, m[0], m[1], m[2], vc)          # the training call, encoder as argument
    arms = {}
    for arm, pat in RUNS.items():                    # build and compile every arm first
        a = A.ACfg(arm=arm, depth=depth, lr=3e-3); k = A.ARMS[arm][0]
        pred = eqx.tree_deserialise_leaves(os.path.join(SA, pat.format(d=depth), "predictor.eqx"), A.make_predictor(vc, a))
        opt, _ = A.optimiser(a, len(Xtr) // a.batch); ost = opt.init(eqx.filter(pred, eqx.is_inexact_array)); stepf = A.make_step(a, opt)
        ctx, Y = sync(enc_call())
        gs = tuple(float(x) for x in sync(A.gain_stats(pred, ctx, *m))) if k is not None else (0.0, 0.0, 0.0)
        rho = 1.0 if (k is None or arm in A.SPC) else max(1.0, gs[1] / k)
        st = (pred, ost)
        for _ in range(3):
            st = sync(stepf(*st, ctx, *m, Y, jnp.float32(rho))[:2])
        arms[arm] = {"a": a, "k": k, "stepf": stepf, "st": st, "rho": rho, "gain": gs, "ts": [], "te": [], "tg": [], "rounds": []}
    order = list(RUNS)
    for rd in range(ROUNDS):                         # rotated order, so drift does not favour one arm
        for arm in order[rd % len(order):] + order[:rd % len(order)]:
            z = arms[arm]; before = gpu_state(); ts = []
            for _ in range(N):
                t0 = time.perf_counter(); c_, y_ = sync(enc_call()); t1 = time.perf_counter()
                z["st"] = sync(z["stepf"](*z["st"], c_, *m, y_, jnp.float32(z["rho"]))[:2]); t2 = time.perf_counter()
                z["te"].append(t1 - t0); ts.append(t2 - t1)
            if z["k"] is not None:
                for _ in range(2):
                    t0 = time.perf_counter(); sync(A.gain_stats(z["st"][0], c_, *m)); z["tg"].append(time.perf_counter() - t0)
            z["ts"] += ts; z["rounds"].append({"round": rd, "update_s_median": float(np.median(ts)), "before": before, "after": gpu_state()})
    for arm, z in arms.items():
        enc_s, step_s = float(np.median(z["te"])), float(np.median(z["ts"])); g_s = float(np.median(z["tg"])) if z["tg"] else 0.0
        row = {"arm": arm, "depth": depth, "batch": z["a"].batch, "T": z["a"].T, "gn_iters": z["a"].gn_iters, "hv_iters": z["a"].hv_iters,
               "encode_s": enc_s, "update_s": step_s, "update_s_q10_q90": [float(np.quantile(z["ts"], .1)), float(np.quantile(z["ts"], .9))],
               "gain_call_s": g_s, "per_step_s": enc_s + step_s + g_s / 50, "rho": z["rho"], "gain_at_final_weights": z["gain"], "rounds": z["rounds"]}
        res["rows"].append(row); print("ROW", json.dumps({x: row[x] for x in ("arm", "depth", "encode_s", "update_s", "gain_call_s", "per_step_s")}), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
for depth in (10, 6):
    rows = [r for r in res["rows"] if r["depth"] == depth]; b = next(r for r in rows if r["arm"] == "bp")
    for r in rows:
        r["x_bp"] = r["per_step_s"] / b["per_step_s"]
        r["x_bp_per_round"] = [x["update_s_median"] / y["update_s_median"] for x, y in zip(r["rounds"], b["rounds"])]
res["co_tenant_seen"] = any(x.get("before", {}).get("n_compute_apps", 0) > 1 or x.get("after", {}).get("n_compute_apps", 0) > 1
                            for r in res["rows"] for x in r["rounds"])
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", [(r["arm"], r["depth"], round(r["x_bp"], 1), [round(v, 1) for v in r["x_bp_per_round"]]) for r in res["rows"]], "co-tenant", res["co_tenant_seen"])
