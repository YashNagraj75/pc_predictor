"""Choose beta (case B) and gamma (case C) by the pre-registered rule D5.

    ./.venv/bin/python select_sparse.py --out sparse_selected.json

Rule (spec: 'Method - Sparse PC-ALM arms spec', Hyperparameters):
  * TRAINING images only, deterministic centre views, never labels or test data.
  * At initialisation (seed 0), target active fraction 0.2 at the sparse variable:
      B:  beta  = rho * quantile(|f_s|, 0.8)          (closed form)
      C:  beta_g = rho = 1 fixed a priori (mu = 1/2); gamma by bisection on
          log(gamma) so that the active fraction of a^(T_test) is 0.2.
The pilot then checks the fraction is still inside [0.1, 0.3] after 2000 steps;
nothing here is re-tuned after seeing a pilot.
"""
import argparse
import dataclasses
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

import data as D
import sparse as SP
from config import RunCfg, SparseCfg

TARGET = 0.2


def centre_views(cfg, n=1024, seed=0):
    Xtr = D.load_dataset(cfg.data)[0]
    idx = np.random.default_rng(seed).choice(len(Xtr), min(n, len(Xtr)), replace=False)
    return D.eval_batch(jnp.asarray(Xtr[idx]), cfg.data)


def select_beta(x, layer):
    cfg = RunCfg(sparse=SparseCfg(case="B", layer=layer, beta=0.0))
    params = SP.init_params(jax.random.PRNGKey(0), cfg, cfg.data.input_dim)
    _, ex = SP.deploy(params, x, cfg)
    beta = cfg.pc.rho * float(jnp.quantile(jnp.abs(ex["f_s"]), 1.0 - TARGET))
    cfg = dataclasses.replace(cfg, sparse=dataclasses.replace(cfg.sparse, beta=beta))
    acts, _ = SP.deploy(params, x, cfg)
    frac = float(jnp.mean(acts[layer] != 0))
    return beta, frac


def select_gamma(x, layer, T_test, lo=1e-5, hi=10.0, iters=40, tol=0.005):
    base = RunCfg(sparse=SparseCfg(case="C", layer=layer, gamma=1.0, beta_g=1.0,
                                   T_test=T_test))
    params = SP.init_params(jax.random.PRNGKey(0), base, base.data.input_dim)

    def frac(g):
        cfg = dataclasses.replace(base, sparse=dataclasses.replace(base.sparse, gamma=g))
        _, ex = SP.deploy(params, x, cfg)
        return float(jnp.mean(ex["a"] != 0))

    flo, fhi = frac(lo), frac(hi)
    assert flo > TARGET > fhi, f"bracket fails: frac({lo})={flo}, frac({hi})={fhi}"
    for _ in range(iters):                  # active fraction decreases in gamma
        mid = float(np.sqrt(lo * hi))
        fm = frac(mid)
        if abs(fm - TARGET) < tol:
            return mid, fm
        lo, hi = (mid, hi) if fm > TARGET else (lo, mid)
    mid = float(np.sqrt(lo * hi))
    return mid, frac(mid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sparse_selected.json")
    ap.add_argument("--layer", type=int, default=SparseCfg.layer)
    ap.add_argument("--T-test", type=int, default=SparseCfg.T_test)
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()
    x = centre_views(RunCfg(), args.n)
    beta, fb = select_beta(x, args.layer)
    gamma, fc = select_gamma(x, args.layer, args.T_test)
    out = {"rule": "active fraction 0.2 at init, seed 0, training centre views",
           "layer": args.layer, "T_test": args.T_test, "n_images": int(x.shape[0]),
           "beta": beta, "active_frac_B": fb, "gamma": gamma, "beta_g": 1.0,
           "active_frac_C": fc}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
