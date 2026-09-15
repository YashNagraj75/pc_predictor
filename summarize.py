
import sys, os, json
sys.path.insert(0, "/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
os.chdir("/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")

names = ["bp_sigreg", "bp_nosigreg", "pc_sigreg_T16_coupled", "pc_nosigreg_T16_coupled"]
print("=== per-run final eval ===")
for n in names:
    log = json.load(open(f"runs/{n}/log.json"))
    last_eval_row = [r for r in log if "eval" in r][-1]
    ev = last_eval_row["eval"]["embedding"]
    print(f"{n:28s} step={last_eval_row['step']:5d} loss_ff={last_eval_row['loss_ff']:.4f} "
          f"knn={ev['knn']:.4f} twonn={ev['twonn']:.2f} "
          f"collapse={json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in ev['collapse'].items()})}")

print()
print("=== cross-arm comparisons (matched configs, same held-out batch) ===")
import jax, numpy as np
import config as C, encoder as E, data as D, metrics as M
from train import compare_arms

Xtr, ytr, Xte, yte = D.load_dataset(C.DataCfg())

for sigreg_tag, bp_name, pc_name in [
    ("sigreg", "bp_sigreg", "pc_sigreg_T16_coupled"),
    ("nosigreg", "bp_nosigreg", "pc_nosigreg_T16_coupled"),
]:
    cfg = json.load(open(f"runs/{bp_name}/config.json"))
    cfg_obj = C.RunCfg(arm="bp", use_sigreg=cfg["use_sigreg"])
    input_dim = C.DataCfg().input_dim
    p_bp = E.init(jax.random.PRNGKey(0), cfg_obj.enc, input_dim)
    p_pc = E.init(jax.random.PRNGKey(0), cfg_obj.enc, input_dim)
    import equinox as eqx
    p_bp = eqx.tree_deserialise_leaves(f"runs/{bp_name}/params.eqx", p_bp)
    p_pc = eqx.tree_deserialise_leaves(f"runs/{pc_name}/params.eqx", p_pc)
    cmp = compare_arms(None, p_bp, None, p_pc, Xte, cfg_obj, n=2000)
    print(f"--- {sigreg_tag} ---")
    print(f"  cka_embedding = {cmp['cka_embedding']:.4f}")
    print(f"  cka_hidden    = {[round(x,4) for x in cmp['cka_hidden']]}")
    print(f"  two_sample_embedding = {cmp['two_sample_embedding']}")
print("SUMMARY_DONE")
