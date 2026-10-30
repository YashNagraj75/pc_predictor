import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax
import data as D, metrics as M, sparse as SP
from run_probes import load_run, embed_split

# Backbone gate for a new dataset: the frozen encoder must carry class information well above its own initialisation
# (kNN k=20 cosine on deterministic centre views).  Also reports embedding spread, to catch collapse.
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n-train", type=int, default=10000); ap.add_argument("--rel", type=float, default=0.05)    # calibrate against the Galaxy line printed by the gates stage
ap.add_argument("--report-only", action="store_true")
a = ap.parse_args()
cfg, p = load_run(a.run)
from config import data_preset
assert cfg.data == data_preset(cfg.data.dataset), f"{a.run} was trained with stale data settings; delete it (review N2-6)"
p0 = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)
Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
res = {"run": a.run, "dataset": cfg.data.dataset, "n_train_total": int(len(Xtr)), "n_test": int(len(Xte)),
       "class_counts_train": np.bincount(ytr, minlength=10).tolist()}
for name, params in (("trained", p), ("init", p0)):
    Ztr, _, itr = embed_split(params, cfg, Xtr, a.n_train, seed=0)
    Zte, _, ite = embed_split(params, cfg, Xte, len(Xte), seed=1)
    Ztr, Zte = np.asarray(Ztr), np.asarray(Zte)
    s = np.linalg.svd(Zte - Zte.mean(0), compute_uv=False); pr = s ** 2 / np.sum(s ** 2)
    sd = Zte.std(0)
    res[name] = {"knn": M.knn_accuracy(Ztr, ytr[itr], Zte, yte[ite], k=20), "finite": bool(np.all(np.isfinite(Zte))),
                 "mean_std": float(sd.mean()), "min_over_median_std": float(sd.min() / (np.median(sd) + 1e-12)),
                 "eff_rank": float(np.exp(-np.sum(pr * np.log(pr + 1e-12))))}
xtr = np.asarray(D.eval_batch(jax.numpy.asarray(Xtr[itr]), cfg.data)); xte = np.asarray(D.eval_batch(jax.numpy.asarray(Xte[ite]), cfg.data))
res["raw_pixel_knn"] = M.knn_accuracy(xtr, ytr[itr], xte, yte[ite], k=20)      # ceiling reference, not gated
chance = float(np.max(np.bincount(yte, minlength=10)) / len(yte))
res["chance_majority"] = chance
# pass: class error cut by >= rel of the untrained encoder's (a fixed margin fails near MNIST's ceiling), above chance,
# no collapsed or dead dimensions
t, i0 = res["trained"], res["init"]
res["rel_err_reduction"] = (1 - i0["knn"] - (1 - t["knn"])) / max(1 - i0["knn"], 1e-6)
res["pass"] = bool(t["finite"] and res["rel_err_reduction"] >= a.rel and t["knn"] > chance + 0.05
                   and t["eff_rank"] >= 0.1 * Ztr.shape[1] and t["min_over_median_std"] >= 0.05)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("dataset", "trained", "init", "raw_pixel_knn", "rel_err_reduction", "chance_majority", "pass")}))
sys.exit(0 if (res["pass"] or a.report_only) else 1)
