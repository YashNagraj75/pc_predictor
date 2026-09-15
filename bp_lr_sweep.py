
import sys, os, time, json, dataclasses as dc
sys.path.insert(0, "/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
os.chdir("/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
import config as C
import train as T

lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
t_grid0 = time.time()
for lr in lrs:
    opt = dc.replace(C.OptCfg(), param_lr=lr)
    cfg = C.RunCfg(arm="bp", use_sigreg=True, tag=f"lr{lr:g}", opt=opt)
    print(f"=== RUN {cfg.name} (param_lr={lr}) ===", flush=True)
    t0 = time.time()
    params, log = T.train(cfg, verbose=True)
    dt = time.time() - t0
    print(f"=== DONE {cfg.name} in {dt:.1f}s ===", flush=True)
print(f"SWEEP_DONE total={time.time()-t_grid0:.1f}s", flush=True)
