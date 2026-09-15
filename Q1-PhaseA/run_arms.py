"""Train arms across seeds and SAVE THEIR WEIGHTS, so probes.py can measure them.

    # seed-replication floor for the two baselines (400-epoch budget)
    ./.venv/bin/python run_arms.py --steps 17200 --seeds 0 1 2 --tag seedgrid

WHY THE OVERWRITE GUARD EXISTS
------------------------------
`RunCfg.name` is derived from the arm and its hyperparameters, NOT from the
seed -- so two runs differing only in seed collide on one directory, and a
re-run silently destroys the weights already there.  That has happened once in
this project: a 2000-step re-run landed on `runs/bp_sigreg` and
`runs/pc_sigreg_T16_coupled` and overwrote 400-epoch weights that could not be
regenerated cheaply.  Every seed therefore gets the seed in its `tag`, and this
script REFUSES to write into a directory that already holds a params.eqx
unless --force is passed.  Trained weights are the one artifact here that no
log or figure can reconstruct.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OptCfg, PCCfg, RunCfg
from train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=17200,
                    help="17200 = the 400-epoch budget the e400 rows used")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--arms", nargs="+", default=["bp", "pc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tag", default="",
                    help="suffix for the run name; the seed is appended to it")
    ap.add_argument("--T", type=int, default=16, help="PC inference budget")
    ap.add_argument("--force", action="store_true",
                    help="permit overwriting a run directory that has weights")
    args = ap.parse_args()

    planned = []
    for seed in args.seeds:
        for arm in args.arms:
            tag = f"{args.tag}_s{seed}" if args.tag else f"s{seed}"
            cfg = RunCfg(arm=arm, use_sigreg=True, out_dir=args.out, tag=tag,
                         pc=PCCfg(T=args.T),
                         opt=OptCfg(n_steps=args.steps,
                                    eval_every=args.eval_every, seed=seed))
            planned.append(cfg)

    # Refuse the whole batch before training anything, rather than discovering
    # the collision partway through a multi-hour run.
    clashes = [c.name for c in planned
               if os.path.exists(os.path.join(c.out_dir, c.name, "params.eqx"))]
    if clashes and not args.force:
        print("REFUSING: these run dirs already hold weights:", flush=True)
        for c in clashes:
            print("   ", os.path.join(args.out, c), flush=True)
        print("Pass --force only if you are certain they are disposable.",
              flush=True)
        sys.exit(1)

    print(f"training {len(planned)} runs x {args.steps} steps", flush=True)
    for cfg in planned:
        t0 = time.time()
        print(f"\n########## {cfg.name}  ({args.steps} steps) ##########",
              flush=True)
        train(cfg, verbose=True)
        print(f"########## {cfg.name} done in {time.time()-t0:.0f}s ##########",
              flush=True)


if __name__ == "__main__":
    main()
