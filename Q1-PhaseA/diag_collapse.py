"""Why does PC collapse the sparse arms where backprop does not?  (pilot job 629aa534)

    ./.venv/bin/python diag_collapse.py --steps 300 --eval-every 10 --out runs_diagcollapse

Re-trains the four pilot arms (PC-B, C1-B, PC-C, C2) from the same init and key
stream as the pilots, but evaluates densely and logs, on top of SP.health:

  deployed network, 1024 deterministic training views
    supports        distinct active patterns at the sparse variable
    embed_std/norm  is the embedding input-dependent / alive?
    spread_*        1 - median cosine to the mean direction, for z_{s-1},
                    the raw f_s (before normalisation) and f~_s -- where does the
                    input information disappear?
    relu_on         fraction of phi(z_{s-1}) > 0 (the gate feeding layer s)
  training relaxation, one FIXED batch (PC arms only)
    gap_s           ||z_s^relax - z_s^deploy|| / ||z_s^deploy||  (train/test gap of
                    removing lambda_s)
    relax_active    active fraction of the relaxed sparse variable
    relax_embed     relaxed vs deployed embedding norm
    move_below      per-layer relative movement of z_l, l < s, from the deployed
                    start: does the output drive reach below s at all?
    lam_next        ||lambda_{s+1}|| / ||z_{s+1}||

Diagnostic only: it extends SP.health in this process; no training code changes.
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
import pcalm as PA
import sigreg as S
import sparse as SP
import train as TR
from config import OptCfg, PCCfg, RunCfg, SparseCfg

_health = SP.health
_FIXED = {}


def _spread(v):
    v = np.asarray(v, np.float64)
    m = v.mean(0)
    c = (v @ m) / (np.linalg.norm(v, axis=1) * np.linalg.norm(m) + 1e-12)
    return float(1.0 - np.median(c))


def _fixed_batch(cfg, B):
    if "Zin" not in _FIXED:
        Xtr = D.load_dataset(cfg.data)[0]
        skey = jax.random.fold_in(jax.random.PRNGKey(cfg.opt.seed + 1), 0)
        kb, kv, ka, kg = jax.random.split(skey, 4)
        idx = np.asarray(jax.random.choice(kb, len(Xtr), (B,), replace=False))
        _FIXED["Zin"], _ = D.make_views(kv, jnp.asarray(Xtr[idx]), cfg.data)
        _FIXED["A"] = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
        _FIXED["key"] = kg
    return _FIXED["Zin"], _FIXED["A"], _FIXED["key"]


def make_health(B_relax):
    def health(params, x, cfg):
        out = dict(_health(params, x, cfg))
        if not SP.active(cfg):
            return out
        sp, s = cfg.sparse, cfg.sparse.layer
        acts, ex = SP.deploy(params, x, cfg)
        v = np.asarray(acts[s] if sp.case == "B" else ex["a"])
        z = np.asarray(acts[-1])
        zprev = acts[s - 1]
        sc = SP._scalings(params, x, cfg)
        f_raw = SP._pred(params, sc, s, zprev, None)
        out.update({
            "supports": len({tuple(np.where(r != 0)[0]) for r in v}),
            "embed_std": float(z.std(0).mean()),
            "embed_norm": float(np.linalg.norm(z, axis=1).mean()),
            "spread_zprev": _spread(zprev), "spread_fraw": _spread(f_raw),
            "spread_ftilde": _spread(ex["f_s"]),
            "relu_on": float(jnp.mean(zprev > 0)),
        })
        if cfg.arm == "pc":
            Zin, A, kg = _fixed_batch(cfg, B_relax)
            ra, rd, _ = PA.relax(params, Zin, A, cfg, kg)
            init = [SP.relax_init(params, Zin[w], cfg) for w in range(Zin.shape[0])]
            a0 = [i[0] for i in init]
            nz = lambda q: float(jnp.linalg.norm(q))
            gap = np.mean([nz(ra[w][s] - a0[w][s]) / (nz(a0[w][s]) + 1e-12) for w in range(len(ra))])
            mv = [float(np.mean([nz(ra[w][l] - a0[w][l]) / (nz(a0[w][l]) + 1e-12) for w in range(len(ra))]))
                  for l in range(s)]
            out.update({
                "gap_s": float(gap),
                "relax_active": float(np.mean([jnp.mean(ra[w][s] != 0) for w in range(len(ra))]))
                                if sp.case == "B" else float("nan"),
                "relax_embed": float(np.mean([jnp.mean(jnp.linalg.norm(ra[w][-1], axis=1)) for w in range(len(ra))])),
                "deploy_embed_batch": float(np.mean([jnp.mean(jnp.linalg.norm(a0[w][-1], axis=1)) for w in range(len(ra))])),
                "move_below": mv,
                "lam_next": float(np.mean([nz(rd[w][s + 1]) / (nz(ra[w][s + 1]) + 1e-12) for w in range(len(ra))])),
            })
        return out
    return health


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", default="runs_diagcollapse")
    ap.add_argument("--sel", default="sparse_selected.json")
    ap.add_argument("--B-relax", type=int, default=64)
    ap.add_argument("--cases", nargs="+", default=["B", "C"])
    ap.add_argument("--arms", nargs="+", default=["pc", "bp"])
    ap.add_argument("--json", default="diag_collapse.json")
    args = ap.parse_args()
    sel = json.load(open(args.sel))
    SP.health = make_health(args.B_relax)
    TR.SP.health = SP.health
    res = {}
    for case in args.cases:
        hp = dict(beta=sel["beta"]) if case == "B" else dict(gamma=sel["gamma"])
        for arm in args.arms:
            cfg = RunCfg(arm=arm, use_sigreg=True, out_dir=args.out, tag=f"diagcol{case}",
                         pc=PCCfg(T=16, eta_h=0.0625, alpha=1.0),
                         opt=OptCfg(n_steps=args.steps, eval_every=args.eval_every, seed=0),
                         sparse=SparseCfg(case=case, **hp))
            _FIXED.clear()
            _, log = TR.train(cfg, verbose=False)
            rows = [{"step": r["step"], "knn": r["eval"]["embedding"]["knn"], **r["eval"]["sparse"]}
                    for r in log if isinstance(r.get("eval"), dict)]
            res[f"{case}-{arm}"] = rows
            last = rows[-1]
            print(f"{case}-{arm}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                 for k, v in last.items() if not isinstance(v, list)}), flush=True)
            json.dump(res, open(args.json, "w"))


if __name__ == "__main__":
    main()
