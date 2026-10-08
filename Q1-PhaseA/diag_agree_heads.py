"""PC-D vs BP-D weight-gradient agreement at the PRODUCTION configuration.

    ./.venv/bin/python diag_agree_heads.py runs/<pc_run_dir> [...] --json agree.json [--init]

Review 2026-10-07 (D-M3, D3): the decisive pair is only interpretable if PC-D's
gradient is BP-D's at the real width/depth/T.  For each case-D run (PC arm's
config; --init uses the initialisation instead of the trained weights), on one
real training batch with the job's key stream:

  task      the full objective (task + heads): per-layer cosine and norm ratio
            |g_pc| / |g_bp| for every encoder layer and every dictionary
  heads     the task drive REMOVED (all views = view 0, SIGReg off): the head
            component alone, which is the positive control that the heads reach
            the encoder at the same strength in both arms
  ratio     |g_head| / |g_task| on the encoder (BP): how strongly the heads
            pull at all

Report-only; never gates.
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
from run_probes import load_run


def _cos(a, b):
    a, b = np.ravel(np.asarray(a)), np.ravel(np.asarray(b))
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _n(x):
    return float(jnp.linalg.norm(x))


def batch(cfg, Xtr, step=0):
    skey = jax.random.fold_in(jax.random.PRNGKey(cfg.opt.seed + 1), step)
    kb, kv, ka, kg = jax.random.split(skey, 4)
    idx = jax.random.choice(kb, len(Xtr), (cfg.opt.batch_size,), replace=False)
    Zin, _ = D.make_views(kv, jnp.asarray(Xtr[np.asarray(idx)]), cfg.data)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
    return Zin, A, kg


def compare(params, cfg, Zin, A, kg):
    cp, cb = dataclasses.replace(cfg, arm="pc"), dataclasses.replace(cfg, arm="bp")
    gp, aux = PA.grads(params, Zin, A, cp, kg)
    gb, _ = TR.bp_grads(params, Zin, A, cb)
    L = len(params[0])
    W = lambda g, l: g[0][l].layers[1].weight
    return {"enc_cos": [_cos(W(gp, l), W(gb, l)) for l in range(L)],
            "enc_ratio": [_n(W(gp, l)) / (_n(W(gb, l)) + 1e-30) for l in range(L)],
            "D_cos": [_cos(a, b) for a, b in zip(gp[2], gb[2])],
            "D_ratio": [_n(a) / (_n(b) + 1e-30) for a, b in zip(gp[2], gb[2])],
            "head_resid": float(aux.get("head_resid", float("nan")))}


def analyse(run_dir, init=False, step=0):
    cfg, params = load_run(run_dir)
    assert cfg.sparse.case == "D", run_dir
    if init:
        params = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)
    Xtr = D.load_dataset(cfg.data)[0]
    if init:
        params = SP.init_mu(params, SP.mu_rows(Xtr, cfg), cfg)
    Zin, A, kg = batch(cfg, Xtr, step)
    out = {"run": run_dir, "init": init, "heads": list(cfg.sparse.heads)}
    out["task"] = compare(params, cfg, Zin, A, kg)
    c0 = dataclasses.replace(cfg, use_sigreg=False)
    Z0 = jnp.stack([Zin[0]] * Zin.shape[0])
    out["heads_only"] = compare(params, c0, Z0, A, kg)
    # BP head vs task magnitude on the encoder
    cb = dataclasses.replace(cfg, arm="bp")
    import equinox as eqx
    g_task = eqx.filter_grad(lambda p: TR.loss_from_views(p, Zin, A, cb)[0])(params)
    g_head = eqx.filter_grad(lambda p: sum(SP.deployed_head_energy(p, Zin[v], cb)
                                           for v in range(Zin.shape[0])))(params)
    W = lambda g, l: g[0][l].layers[1].weight
    out["head_over_task_enc"] = [_n(W(g_head, l)) / (_n(W(g_task, l)) + 1e-30) for l in range(len(params[0]))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--json", default="agree_heads.json")
    args = ap.parse_args()
    res = []
    for rd in args.runs:
        r = analyse(rd, init=args.init)
        res.append(r)
        mh = max(r["heads"])
        f = lambda v: " ".join(f"{x:.2f}" for x in v)
        print(f"{rd} init={args.init}  head_resid={r['task']['head_resid']:.4f}\n"
              f"   task : enc cos {f(r['task']['enc_cos'])} | ratio {f(r['task']['enc_ratio'])} | D cos {f(r['task']['D_cos'])}\n"
              f"   heads: enc cos(<=max head) {f(r['heads_only']['enc_cos'][:mh + 1])} | ratio {f(r['heads_only']['enc_ratio'][:mh + 1])}"
              f" | D cos {f(r['heads_only']['D_cos'])} ratio {f(r['heads_only']['D_ratio'])}\n"
              f"   |g_head|/|g_task| enc {f(r['head_over_task_enc'])}", flush=True)
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
