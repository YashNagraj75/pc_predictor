"""Re-train the two SIGReg arms and SAVE THEIR WEIGHTS, so probes.py has
something to measure.

The historical Q1 grids kept only logs and figures, not params.eqx, so the
latent-quality suite cannot be applied to them retrospectively.  This script
reproduces the 2000-step post-fix budget (the cheap grid, not the 400-epoch
one) for the two arms that matter for the representation question.

    ./.venv/bin/python run_arms.py --steps 2000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OptCfg, RunCfg
from train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--arms", nargs="+", default=["bp", "pc"])
    args = ap.parse_args()

    for arm in args.arms:
        cfg = RunCfg(arm=arm, use_sigreg=True, out_dir=args.out,
                     opt=OptCfg(n_steps=args.steps, eval_every=args.eval_every))
        t0 = time.time()
        print(f"\n########## {cfg.name}  ({args.steps} steps) ##########", flush=True)
        train(cfg, verbose=True)
        print(f"########## {cfg.name} done in {time.time()-t0:.0f}s ##########",
              flush=True)


if __name__ == "__main__":
    main()
