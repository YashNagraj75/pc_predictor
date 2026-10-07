"""Pre-registered choices for case D (side dictionary heads), all K at once.

    ./.venv/bin/python select_heads.py --out heads_selected.json

Spec: wiki 'Method - Sparse side-head arm plan' (user decisions 2026-10-07).
  * Nested head sets growing outward from layer 5:
        K=1 {5}, K=2 {4,5}, K=3 {4,5,6}, K=4 {3,4,5,6}, K=5 {3,4,5,6,7}.
  * gamma_l per head: bisection on log(gamma) so that a_l^(T_test) is 20 % active
    at INITIALISATION, seed 0, deterministic centre views of TRAINING images only.
    The encoder is the unchanged original and D_l is drawn from a fold keyed by l,
    so gamma_l is the same in every head set that contains l.
  * Head weights, FIXED TOTAL split equally: for each K, every head contributes the
    same share of a total head energy equal to 0.25 x the task loss at init, as the
    BP-D objective counts it (head energy summed over the V views, task loss once):
        V * w_l * e_l = 0.25 * L_task / K,   e_l = unweighted per-view head energy.
Nothing here is re-tuned after a pilot.
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

import data as D
import encoder as E
import sigreg as S
import sparse as SP
import train as TR
from config import PCCfg, RunCfg, SparseCfg

ORDER = (5, 4, 6, 3, 7)                 # outward from 5; K heads = first K, sorted
TARGET, TOTAL_RATIO = 0.2, 0.25
BETA_G, T_TEST = 1.0, 16
PC = PCCfg(T=16, eta_h=0.0625, alpha=1.0)


def head_sets():
    return {K: tuple(sorted(ORDER[:K])) for K in range(1, len(ORDER) + 1)}


def cfg_for(heads, gammas, ws):
    return RunCfg(arm="bp", pc=PC, sparse=SparseCfg(case="D", heads=tuple(heads), gammas=tuple(gammas),
                                                     head_ws=tuple(ws), beta_g=BETA_G, T_test=T_TEST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="heads_selected.json")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    base = RunCfg()
    Xtr = D.load_dataset(base.data)[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), args.n, replace=False)
    x = D.eval_batch(jnp.asarray(Xtr[idx]), base.data)        # deterministic centre views

    # all five heads at once: encoder identical to case none, D_l keyed by l
    all_heads = tuple(sorted(ORDER))
    cfg_all = cfg_for(all_heads, (1.0,) * 5, (1.0,) * 5)
    params = SP.init_params(jax.random.PRNGKey(0), cfg_all, base.data.input_dim)
    z, hs = E.forward(params, x, base.enc)
    acts = [*hs, z]

    gammas = {}
    for l, Dl in zip(all_heads, SP.get_D(params)):
        f = acts[l]
        eta = cfg_all.sparse.step_frac / (BETA_G * SP.spectral_sq(Dl))

        def frac(g):
            a = SP.ista(f, Dl, g, BETA_G, T_TEST, eta)
            return float(jnp.mean(a != 0))
        lo, hi = 1e-5, 10.0
        for _ in range(40):
            mid = float(np.sqrt(lo * hi)); fm = frac(mid)
            if abs(fm - TARGET) < 0.005:
                break
            lo, hi = (mid, hi) if fm > TARGET else (lo, mid)
        gammas[l] = (mid, fm)
        print(f"head {l}: gamma={mid:.5g}  active={fm:.4f}", flush=True)

    # task loss and per-head energies at init, on a TRAINING batch with real views
    kv, ka = jax.random.split(jax.random.PRNGKey(123))
    Zin, _ = D.make_views(kv, jnp.asarray(Xtr[idx[:args.batch]]), base.data)
    A = S.sample_directions(ka, base.enc.embed_dim, base.sig.num_slices)
    V = Zin.shape[0]
    task, _ = TR.loss_from_views(params, Zin, A, cfg_all)
    task = float(task)
    e = {}
    for l in all_heads:
        one = cfg_for((l,), (gammas[l][0],), (1.0,))
        pl = (params[0], params[1], (SP.get_D(params)[all_heads.index(l)],))
        e[l] = float(sum(SP.deployed_head_energy(pl, Zin[v], one) for v in range(V))) / V
        print(f"head {l}: unweighted per-view head energy e={e[l]:.5g}", flush=True)

    sets = {}
    for K, hset in head_sets().items():
        ws = [TOTAL_RATIO * task / (K * V * e[l]) for l in hset]
        tot = V * sum(w * e[l] for w, l in zip(ws, hset))
        sets[str(K)] = {"heads": list(hset), "gammas": [gammas[l][0] for l in hset],
                        "head_ws": ws, "active_at_init": [gammas[l][1] for l in hset],
                        "head_energy_total_at_init": tot, "ratio_to_task": tot / task}
    out = {"rule": "nested outward from 5; gamma_l -> 20% active at init (seed 0, training centre "
                   "views); fixed total head energy 0.25 x task loss at init, split equally",
           "beta_g": BETA_G, "T_test": T_TEST, "task_loss_at_init": task, "n_views": V,
           "n_images": int(x.shape[0]), "per_head_energy": {str(k): v for k, v in e.items()},
           "sets": sets}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else [round(q, 5) for q in vv])
                          for kk, vv in v.items()} for k, v in sets.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
