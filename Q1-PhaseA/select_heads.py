"""Pre-registered choices for case D (side dictionary heads), all K at once.

    ./.venv/bin/python select_heads.py --out heads_selected.json

Spec: wiki 'Method - Sparse side-head arm plan' (user decisions 2026-10-07).
  * Nested head sets growing outward from layer 5:
        K=1 {5}, K=2 {4,5}, K=3 {4,5,6}, K=4 {3,4,5,6}, K=5 {3,4,5,6,7}.
  * Heads explain the normalised activity f~_l = sqrt(n) f_l/||f_l|| (sparse.head_in).
  * gamma_l per head: bisection on log(gamma) so that a_l^(T_test) is 20 % active
    at INITIALISATION, seed 0, deterministic centre views of TRAINING images only.
    The encoder is the unchanged original and D_l is drawn from a fold keyed by l,
    so gamma_l is the same in every head set that contains l.
  * Head weights, FIXED TOTAL split equally, calibrated by GRADIENT (user decision
    2026-10-07, after the energy calibration gave a head pull of only 2-6 % of the
    task gradient on the encoder):
      - split: within a K, heads contribute equal ENERGY (w_l proportional to 1/e_l,
        e_l = unweighted per-view head energy as the BP-D objective counts it);
      - scale: one factor c_K so that, at init on one training batch,
            || dHead/dW_{0..max head} || = 0.25 || dTask/dW_{0..max head} ||
        (BP gradients, encoder weight matrices of layers 0 .. max head).
    The gradient is linear in the weights, so c_K is exact from one measurement.
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
TARGET, GRAD_RATIO = 0.2, 0.25
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
        f = SP.head_in(acts[l])                 # heads explain the normalised copy
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
        assert abs(fm - TARGET) < 0.01, f"head {l}: bisection ended at {fm:.4f} active (gamma {mid:.3g})"
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

    import equinox as eqx
    W = lambda g, l: g[0][l].layers[1].weight
    g_task = eqx.filter_grad(lambda p: TR.loss_from_views(p, Zin, A, cfg_all)[0])(params)

    def enc_norm(g, top):
        return float(jnp.sqrt(sum(jnp.sum(W(g, l) ** 2) for l in range(top + 1))))

    sets = {}
    for K, hset in head_sets().items():
        base = [1.0 / (K * V * e[l]) for l in hset]            # equal energy per head
        cK = cfg_for(hset, [gammas[l][0] for l in hset], base)
        pK = (params[0], params[1], tuple(SP.get_D(params)[all_heads.index(l)] for l in hset))
        g_head = eqx.filter_grad(lambda p: sum(SP.deployed_head_energy(p, Zin[v], cK)
                                               for v in range(V)))(pK)
        top = max(hset)
        r0 = enc_norm(g_head, top) / enc_norm(g_task, top)
        c = GRAD_RATIO / r0
        ws = [c * b for b in base]
        tot = V * sum(w * e[l] for w, l in zip(ws, hset))
        sets[str(K)] = {"heads": list(hset), "gammas": [gammas[l][0] for l in hset],
                        "head_ws": ws, "active_at_init": [gammas[l][1] for l in hset],
                        "grad_ratio_at_init": GRAD_RATIO, "grad_ratio_unit_weights": r0,
                        "head_energy_total_at_init": tot, "energy_ratio_to_task": tot / task}
        print(f"K={K} heads={hset}: scale {c:.3g} -> energy {tot / task:.3g} x task", flush=True)
    out = {"rule": "nested outward from 5; gamma_l -> 20% active at init (seed 0, training centre "
                   "views); equal energy per head within K, total scaled so the head gradient on encoder "
                   "layers <= max head is 0.25 x the task gradient at init (BP, one training batch)",
           "beta_g": BETA_G, "T_test": T_TEST, "task_loss_at_init": task, "n_views": V,
           "n_images": int(x.shape[0]), "per_head_energy": {str(k): v for k, v in e.items()},
           "sets": sets}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else [round(q, 5) for q in vv])
                          for kk, vv in v.items()} for k, v in sets.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
