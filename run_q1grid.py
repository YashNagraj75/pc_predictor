
import sys, os, time, json
sys.path.insert(0, "/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
os.chdir("/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
import config as C
import train as T

t_grid0 = time.time()
for cfg in C.q1_grid():
    print(f"=== RUN {cfg.name} ===", flush=True)
    t0 = time.time()
    params, log = T.train(cfg, verbose=True)
    dt = time.time() - t0
    print(f"=== DONE {cfg.name} in {dt:.1f}s ===", flush=True)
print(f"GRID_DONE total={time.time()-t_grid0:.1f}s", flush=True)
