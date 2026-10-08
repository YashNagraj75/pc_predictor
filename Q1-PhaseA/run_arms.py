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

from config import OptCfg, PCCfg, RunCfg, SparseCfg
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
    ap.add_argument("--eta-h", type=float, default=None,
                    help="PER-SAMPLE activity step; pcalm.relax multiplies by "
                         "the batch size.  Set it from diag_inference.py at "
                         "the batch size you are actually training at.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="PC-ALM dual rate; 0 = plain PC")
    # ---- sparse prior (spec: 'Method - Sparse PC-ALM arms spec') ----------
    ap.add_argument("--sparse-case", choices=["none", "B", "C", "D"], default="none")
    ap.add_argument("--sparse-layer", type=int, default=SparseCfg.layer)
    ap.add_argument("--beta", type=float, default=0.0, help="case B L1 weight")
    ap.add_argument("--gamma", type=float, default=0.0, help="case C L1 weight on a")
    ap.add_argument("--beta-g", type=float, default=SparseCfg.beta_g)
    ap.add_argument("--dict-mult", type=int, default=SparseCfg.dict_mult)
    ap.add_argument("--T-test", type=int, default=SparseCfg.T_test)
    ap.add_argument("--below-soft", action="store_true",
                    help="drop the multipliers below s too (D3 alternative)")
    # ---- case D: side heads (wiki 'Method - Sparse side-head arm plan') -----
    ap.add_argument("--heads-json", default=None,
                    help="case D: JSON from select_heads.py; with --K picks the nested "
                         "head set, its gammas and its head weights")
    ap.add_argument("--K", type=int, default=None, help="case D: number of heads")
    ap.add_argument("--init-only", action="store_true",
                    help="write config.json + params.eqx at INITIALISATION and "
                         "stop: the untrained floor for this architecture")
    ap.add_argument("--force", action="store_true",
                    help="permit overwriting a run directory that has weights")
    args = ap.parse_args()
    if args.sparse_case != "none" and not args.tag:
        # RunCfg.name does not encode the sparse settings; only the tag
        # separates these runs, so an untagged sparse run would collide.
        ap.error("--tag is required with --sparse-case B/C/D")
    if args.sparse_case != "none" and (args.alpha is None or args.eta_h is None
                                       or args.alpha <= 0):
        # the defaults (alpha=0, eta_h=0.25) would silently run PLAIN PC at the
        # wrong step and the run name would not show it (found in review)
        ap.error("--sparse-case B/C/D needs explicit --alpha > 0 and --eta-h "
                 "(spec operating point: --alpha 1 --eta-h 0.0625)")
    if args.eta_h is None:
        args.eta_h = PCCfg.eta_h
    if args.alpha is None:
        args.alpha = 0.0
    if args.sparse_case == "B" and args.beta <= 0:
        ap.error("case B needs --beta > 0")
    if args.sparse_case == "C" and args.gamma <= 0:
        ap.error("case C needs --gamma > 0")
    D_kw = {}
    if args.sparse_case == "D":
        if not args.heads_json or not args.K:
            ap.error("case D needs --heads-json and --K")
        import json as _json
        sel = _json.load(open(args.heads_json))
        k = str(args.K)
        if k not in sel["sets"]:
            ap.error(f"K={args.K} not in {args.heads_json} (has {sorted(sel['sets'])})")
        st = sel["sets"][k]
        D_kw = dict(heads=tuple(st["heads"]), gammas=tuple(st["gammas"]),
                    head_ws=tuple(st["head_ws"]), beta_g=sel["beta_g"], T_test=sel["T_test"],
                    beta_r=sel.get("beta_r", 0.0), code_k=sel.get("code_k", 1))
        if args.beta_g != SparseCfg.beta_g or args.T_test != SparseCfg.T_test:
            ap.error("case D takes beta_g and T_test from --heads-json only")
        if args.T != sel["T_test"]:
            # the relaxed code takes a0 + T steps and the deployed one a0 + T_test:
            # they are the same quantity only if T == T_test (review D-M9)
            ap.error(f"case D needs --T == T_test ({sel['T_test']})")
    sparse = SparseCfg(case=args.sparse_case, layer=args.sparse_layer,
                       beta=args.beta, gamma=args.gamma,
                       beta_g=D_kw.pop("beta_g", args.beta_g),
                       dict_mult=args.dict_mult, T_test=D_kw.pop("T_test", args.T_test),
                       below_hard=not args.below_soft, **D_kw)

    planned = []
    for seed in args.seeds:
        for arm in args.arms:
            tag = f"{args.tag}_s{seed}" if args.tag else f"s{seed}"
            cfg = RunCfg(arm=arm, use_sigreg=True, out_dir=args.out, tag=tag,
                         pc=PCCfg(T=args.T, eta_h=args.eta_h,
                                  alpha=args.alpha),
                         opt=OptCfg(n_steps=args.steps,
                                    eval_every=args.eval_every, seed=seed),
                         sparse=sparse)
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

    if args.init_only:
        import jax, equinox as eqx, sparse as SP
        for cfg in planned:
            out = os.path.join(cfg.out_dir, cfg.name)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "config.json"), "w") as f:
                f.write(cfg.to_json())
            p = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg,
                               cfg.data.input_dim)
            import data as Dd
            p = SP.init_mu(p, SP.mu_rows(Dd.load_dataset(cfg.data)[0], cfg), cfg)
            eqx.tree_serialise_leaves(os.path.join(out, "params.eqx"), p)
            print("init-only:", out, flush=True)
        return

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
