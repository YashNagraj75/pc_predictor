import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax
import data as D, metrics as M, sparse as SP
from run_probes import load_run, embed_split

# Backbone gate for a new dataset: the frozen encoder must carry class information well above its own initialisation
# (kNN k=20 cosine on deterministic centre views).  Also reports embedding spread, to catch collapse.
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n-train", type=int, default=10000); ap.add_argument("--margin", type=float, default=0.05)
a = ap.parse_args()
cfg, p = load_run(a.run)
p0 = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)
Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
res = {"run": a.run, "dataset": cfg.data.dataset, "n_train_total": int(len(Xtr)), "n_test": int(len(Xte)),
       "class_counts_train": np.bincount(ytr, minlength=10).tolist()}
for name, params in (("trained", p), ("init", p0)):
    Ztr, _, itr = embed_split(params, cfg, Xtr, a.n_train, seed=0)
    Zte, _, ite = embed_split(params, cfg, Xte, len(Xte), seed=1)
    Ztr, Zte = np.asarray(Ztr), np.asarray(Zte)
    s = np.linalg.svd(Zte - Zte.mean(0), compute_uv=False); pr = s ** 2 / np.sum(s ** 2)
    res[name] = {"knn": M.knn_accuracy(Ztr, ytr[itr], Zte, yte[ite], k=20), "finite": bool(np.all(np.isfinite(Zte))),
                 "mean_std": float(Zte.std(0).mean()), "eff_rank": float(np.exp(-np.sum(pr * np.log(pr + 1e-12))))}
chance = float(np.max(np.bincount(yte, minlength=10)) / len(yte))
res["chance_majority"] = chance
res["pass"] = bool(res["trained"]["finite"] and res["trained"]["knn"] > res["init"]["knn"] + a.margin
                   and res["trained"]["knn"] > chance + a.margin and res["trained"]["eff_rank"] > 5)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("dataset", "trained", "init", "chance_majority", "pass")}))
sys.exit(0 if res["pass"] else 1)
