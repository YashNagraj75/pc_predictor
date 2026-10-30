import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, pred_eval as PE
from run_probes import load_run

# Prep-time check (review N6/P7): the stored test latents must be reproduced by re-encoding the test views exactly as
# the robustness tests will, and the latents must come from the backbone now on disk (sha256 stamp).
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--latents", default="pred_latents.npz")
ap.add_argument("--stamp", default="pred_latents.backbone_sha256")
a = ap.parse_args()
sha = hashlib.sha256(open(os.path.join(a.run, "params.eqx"), "rb").read()).hexdigest()
if os.path.exists(a.stamp):
    assert open(a.stamp).read().strip() == sha, "pred_latents.npz was made by a different backbone; delete it and re-extract"
cfg, p = load_run(a.run)
L = np.load(a.latents)
assert str(L["backbone_run"]) == a.run, (str(L["backbone_run"]), a.run)
_, ytr, Xte, yte = D.load_dataset(cfg.data)
assert np.array_equal(L["yte"], yte) and np.array_equal(L["ytr"], ytr[:len(L["ytr"])]), "label order differs from the loader"
z = PE.encode(p, cfg.enc, PE.test_views(cfg.data, np.asarray(Xte)))
err = PE.check_reproduction(z, L["bp_emb_te"])
for k in ("bp_emb_tr", "bp_emb_te"):
    assert np.all(np.isfinite(L[k])), k
open(a.stamp, "w").write(sha)
print(json.dumps({"reproduction_rel_err": err, "n_train": int(len(L["ytr"])), "n_test": int(len(L["yte"])), "backbone_sha256": sha[:16]}))
