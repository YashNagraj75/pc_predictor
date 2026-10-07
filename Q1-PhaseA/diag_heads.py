"""Case D analysis: are the sparse side-head codes useful and robust?

    ./.venv/bin/python diag_heads.py runs/<run> [...] --csv diag_heads.csv

Spec: wiki 'Method - Sparse side-head arm plan', Analysis.  For every run:

  embedding   linear (closed-form ridge) and kNN probes, clean and under test-time
              corruption (Gaussian noise, random-square occlusion; shared seed).
              Runs on reference arms too (case none: PC-ALM, BP), one protocol.
  codes       (case D, and case-none runs given --heads-json, which attaches the
              K=5 heads' dictionaries at their INITIAL draw so the code probe has an
              untrained-dictionary reference on a trained encoder)
              per head l, a_l^(t) for t in {0, T_test}: probes on each code and on
              the concatenation of all heads' codes; active fraction; distinct
              supports; relative reconstruction error; dead atoms.
              delta_inf = probe(a^(T)) - probe(a^(0)) per condition.
  spread      input spread (1 - median cosine to the mean) of every layer's
              activity and of the embedding -- does stacking priors erase input
              variation as cases B and C did?

Probes are fit on CLEAN training features and evaluated on (possibly corrupted)
test features.  Corruption is applied only at test time.
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

import data as D
import encoder as E
import sparse as SP
from config import SparseCfg
from diag_sparse import NOISE, OCCL, batched, corrupt, probe_pair
from run_probes import load_run


def _spread(v):
    v = np.asarray(v)
    m = v.mean(0)
    c = (v @ m) / (np.linalg.norm(v, axis=1) * np.linalg.norm(m) + 1e-12)
    return float(1.0 - np.median(c))


def attach_init_heads(cfg, params, heads_json):
    """Case-none run + the K=5 head set at the dictionaries' INITIAL draw."""
    sel = json.load(open(heads_json))
    st = sel["sets"]["5"]
    sp = SparseCfg(case="D", heads=tuple(st["heads"]), gammas=tuple(st["gammas"]),
                   head_ws=tuple(st["head_ws"]), beta_g=sel["beta_g"], T_test=sel["T_test"])
    c2 = dataclasses.replace(cfg, sparse=sp)
    fresh = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), c2, cfg.data.input_dim)
    return c2, (params[0], params[1], fresh[2])


def readout(params, x, cfg, t):
    """-> [embedding, all activities stacked per layer..., codes per head...]"""
    if cfg.sparse.case == "D":
        acts, ex = SP.deploy(params, x, cfg, t=t)
        return [acts[-1]] + list(acts) + list(ex["a"])
    z, hs = E.forward(params, x, cfg.enc)
    return [z] + [*hs, z]


def analyse(run_dir, n_train, n_test, heads_json=None):
    cfg, params = load_run(run_dir)
    tag = os.path.basename(run_dir.rstrip("/"))
    if cfg.sparse.case == "none" and heads_json:
        cfg, params = attach_init_heads(cfg, params, heads_json)
        tag += "+init_heads"
    sp = cfg.sparse
    Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
    rtr = np.random.default_rng(0).choice(len(Xtr), min(n_train, len(Xtr)), replace=False)
    rte = np.random.default_rng(1).choice(len(Xte), min(n_test, len(Xte)), replace=False)
    xtr = D.eval_batch(jnp.asarray(Xtr[rtr]), cfg.data)
    xte = D.eval_batch(jnp.asarray(Xte[rte]), cfg.data)
    ytr_, yte_ = np.asarray(ytr)[rtr], np.asarray(yte)[rte]
    L = len(params[0])
    K = len(sp.heads) if sp.case == "D" else 0
    rows = []

    def add(metric, value, condition="clean", t="", head=""):
        rows.append({"run": tag, "case": sp.case, "arm": cfg.arm, "seed": cfg.opt.seed,
                     "K": K, "metric": metric, "condition": condition, "t": t,
                     "head": head, "value": value})

    ts = [("0", 0), ("T", sp.T_test)] if sp.case == "D" else [("ff", None)]
    conds = [("clean", 0.0)] + [("noise", s) for s in NOISE] + [("occlusion", o) for o in OCCL]
    ftr = {lab: batched(lambda xb, t=t: readout(params, xb, cfg, t), xtr) for lab, t in ts}

    # spread of every layer (clean test) -- from the deployed / feedforward pass
    te_clean = batched(lambda xb: readout(params, xb, cfg, ts[-1][1]), xte)
    for l in range(L):
        add("input_spread", _spread(te_clean[1 + l]), head=str(l))
    add("embed_spread", _spread(te_clean[0]))
    add("embed_norm", float(np.linalg.norm(te_clean[0], axis=1).mean()))

    for kind, lvl in conds:
        xc = corrupt(xte, cfg, kind, lvl)
        cname = kind if kind == "clean" else f"{kind}={lvl:g}"
        got = {}
        for lab, t in ts:
            fte = batched(lambda xb, t=t: readout(params, xb, cfg, t), xc)
            for k, v in probe_pair(ftr[lab][0], ytr_, fte[0], yte_).items():
                add(f"embed_{k}", v, cname, lab)
            if sp.case != "D":
                continue
            codes_tr = ftr[lab][1 + L:]
            codes_te = fte[1 + L:]
            for l, ctr, cte in zip(sp.heads, codes_tr, codes_te):
                pr = probe_pair(ctr, ytr_, cte, yte_)
                got[(lab, l)] = pr
                for k, v in pr.items():
                    add(f"code_{k}", v, cname, lab, str(l))
                if kind == "clean":
                    on = cte != 0
                    add("code_active", float(on.mean()), cname, lab, str(l))
                    add("code_supports", len({r.tobytes() for r in on}), cname, lab, str(l))
                    add("code_dead_atoms", float(np.mean(~on.any(0))), cname, lab, str(l))
            cat_tr, cat_te = np.concatenate(codes_tr, 1), np.concatenate(codes_te, 1)
            pr = probe_pair(cat_tr, ytr_, cat_te, yte_)
            got[(lab, "all")] = pr
            for k, v in pr.items():
                add(f"code_{k}", v, cname, lab, "all")
        if sp.case == "D":
            for h in list(sp.heads) + ["all"]:
                for k in ("linear", "knn"):
                    add(f"delta_inf_code_{k}", got[("T", h)][k] - got[("0", h)][k], cname, "T-0", str(h))
    if sp.case == "D":
        acts, ex = SP.deploy(params, xte[:1000], cfg)
        for l, Dl, a in zip(sp.heads, SP.get_D(params), ex["a"]):
            f = acts[l]
            add("code_rel_recon", float(jnp.mean(jnp.linalg.norm(f - a @ Dl.T, axis=-1)
                                                 / (jnp.linalg.norm(f, axis=-1) + 1e-12))), head=str(l))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", default="diag_heads.csv")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--heads-json", default=None,
                    help="attach initial-draw K=5 heads to case-none runs (code reference)")
    args = ap.parse_args()
    allr = []
    for rd in args.runs:
        if not os.path.exists(os.path.join(rd, "params.eqx")):
            print(f"[skip] {rd}: no params.eqx", flush=True); continue
        if os.path.exists(os.path.join(rd, "DIVERGED")):
            print(f"[skip] {rd}: DIVERGED", flush=True); continue
        print(f"=== {rd}", flush=True)
        rows = analyse(rd, args.n_train, args.n_test, args.heads_json)
        allr += rows
        for r in rows:
            if r["condition"] in ("clean", "noise=0.2") and r["metric"] in ("embed_linear", "code_linear") \
                    and r["head"] in ("", "all"):
                print(f"   {r['metric']:<13} {r['condition']:<10} t={r['t']:<3} {r['value']:.4f}", flush=True)
        with open(args.csv, "w", newline="") as f:          # rewritten after every run
            w = csv.DictWriter(f, fieldnames=list(allr[0].keys()))
            w.writeheader(); w.writerows(allr)
    print(f"wrote {args.csv} ({len(allr)} rows)", flush=True)


if __name__ == "__main__":
    main()
