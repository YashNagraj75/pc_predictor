import glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
# GPU-vs-CPU gradient gate at real training shapes and trained weights (findings/41: Triton GEMMs gave wrong batched
# gradients).  run <root> <tag> <outdir> [depth]: save flat gradients under the current backend/flags.
# compare <outdir>: every GPU tag must match the CPU reference to rel < 1e-4; exit 1 otherwise.
if sys.argv[1] == "compare":
    d, bad, rows = sys.argv[2], 0, []
    for ref in sorted(glob.glob(f"{d}/cpu_*.npy")):
        name = os.path.basename(ref)[4:-4]; r = np.load(ref)
        for f in sorted(glob.glob(f"{d}/*_{name}.npy")):
            tag = os.path.basename(f)[:-len(name) - 5]
            if tag == "cpu": continue
            g = np.load(f); rel = float(np.linalg.norm(g - r) / np.linalg.norm(r)); ok = rel < 1e-4; bad += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {name}: GPU vs CPU rel {rel:.2e}", flush=True)
    print("GPU GATE PASS" if bad == 0 else "GPU GATE FAIL"); sys.exit(1 if bad else 0)
import jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A
ROOT, TAG, OUTD = sys.argv[2:5]; depth = int(sys.argv[5]) if len(sys.argv) > 5 else 10; os.makedirs(OUTD, exist_ok=True)
V0 = os.path.join(ROOT, "Q1-PhaseA/runs_vit/v0_d6_s0"); vc, enc, tenc, _ = A.load_encoders(V0)
SA = os.path.join(ROOT, "Q1-PhaseA/runs_vit/stageA"); a = A.ACfg(arm="epc_alm_k4", depth=depth)
ck = sorted(glob.glob(f"{SA}/*_d{depth}_*_e25_full_s0*/predictor.eqx")) + sorted(glob.glob(f"{SA}_invalid_f41/epc_alm_k4_d{depth}_*_e25_full_s0*/predictor.eqx"))
pred = eqx.tree_deserialise_leaves(ck[0], A.make_predictor(vc, a)) if ck else A.make_predictor(vc, a)
Xtr, _, _, _ = D.load_cifar10(root=os.path.join(ROOT, "data"), n_train=-1, n_test=2000, seed=0)
flat = lambda t: np.concatenate([np.asarray(v).ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])
rng = np.random.default_rng(5); X = Xtr[40000:40064]
ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, 64, vc))
im = jnp.asarray(V.standardise(X)); ctx, Y = A.encode(enc, tenc, im, im, ci, cv, ti, vc); b = (ctx, ci, cv, ti, tv, Y)
X2 = jnp.asarray(V.standardise(Xtr[:256])); m2 = [jnp.asarray(x) for x in V.sample_masks(np.random.default_rng(6), 256, vc)]
e0, p0, _ = V.init(vc)                            # V0 loss at batch 256 with the TRAINED encoders and a fresh V0 predictor
fns = {"pred_bp": lambda: P.grads(pred, *b, "bp", chunk=64)[0],
       "pred_epc": lambda: P.grads(pred, *b, "epc", rho=500.0, alpha=1.0, T=10, chunk=32, gn_iters=4)[0],
       "pred_spc_u1": lambda: P.grads(pred, *b, "spc", rho=1.0, alpha=0.0, T=20, chunk=32, hv_iters=8)[0],
       "pred_epc_blk": lambda: P.grads(pred, *b, "epc", rho=1.0, alpha=1.0, T=10, chunk=32, gn_iters=4, rho_rel=4.0)[0],
       "v0_loss": lambda: eqx.filter_grad(V.loss_fn)((enc, p0), tenc, X2, *m2, vc)}
out = {"tag": TAG, "backend": jax.default_backend(), "xla_flags": os.environ.get("XLA_FLAGS", ""), "ckpt": ck[0] if ck else "init"}
for name, f in fns.items():
    g = eqx.filter_jit(f)(); np.save(f"{OUTD}/{TAG}_{name}.npy", flat(g)); print(name, "saved", flush=True)
json.dump(out, open(f"{OUTD}/{TAG}.json", "w")); print("DONE", json.dumps(out))
