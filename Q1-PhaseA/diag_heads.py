"""Case D analysis: are the sparse side-head codes useful and robust?

    ./.venv/bin/python diag_heads.py runs/<run> [...] --heads-json heads_selected.json --csv diag_heads.csv

Spec: wiki 'Method - Sparse side-head arm plan', Analysis.  Every row has a ROLE
(review D-M2/D8/N6), so trained, floor and reference rows can never pool:

  trained_D            a trained case-D run (K = number of heads)
  trained_enc_init_D   the same trained encoder with its dictionaries put back to
                       the initial draw: code(trained D) - code(init D) on the SAME
                       encoder is the headline dictionary-learning effect (D7)
  floor_init           untrained encoder + initial K=5 heads; also probes every nested
                       head subset so K<5 runs have a matched floor (D7)
  reused_headfree      reused PC-ALM / BP runs (K = 0): embedding only
  reused_enc_init_D    the same reused encoders with the initial K=5 heads attached

Metrics: embed_* (embedding, written once), f_* (the head layer's own activity f_l
and the concatenation over heads -- the layer-matched reference for codes, D6),
code_* per head and per concatenation at t = 0 and t = T_test, code_active /
supports / dead atoms, delta_inf_code_* = code(T) - code(0), input_spread and
layer_norm per layer.  Probes are fit on CLEAN training features and evaluated on
(possibly corrupted) test features; codes use a per-feature scale floor of
0.1 x median (D10).  Probe differences on 2000 test rows carry a binomial SE of
about 1 pp -- read small deltas accordingly (D9).
"""
import argparse
import csv

import dataclasses
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import data as D
import encoder as E
import sparse as SP
from config import SparseCfg
from diag_sparse import NOISE, OCCL, corrupt, probe_pair
from run_probes import load_run


CODE_SD_FLOOR = 0.1       # ridge per-feature scale floor for sparse codes (review D10)
ORDER = (5, 4, 6, 3, 7)   # nested head sets, as select_heads.ORDER


def _spread(v):
    v = np.asarray(v)
    m = v.mean(0)
    c = (v @ m) / (np.linalg.norm(v, axis=1) * np.linalg.norm(m) + 1e-12)
    return float(1.0 - np.median(c))


def k5_sparse(heads_json):
    sel = json.load(open(heads_json))
    st = sel["sets"]["5"]
    return SparseCfg(case="D", heads=tuple(st["heads"]), gammas=tuple(st["gammas"]),
                     head_ws=tuple(st["head_ws"]), beta_g=sel["beta_g"], T_test=sel["T_test"],
                     beta_r=sel.get("beta_r", 0.0))


def init_D(cfg):
    """The INITIAL dictionaries of cfg's heads (same key folds as init_params)."""
    return SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)[2]


def with_init_dicts(params, cfg):
    """Same encoder, mu and pixel mean; dictionaries D (and case-E G) at their initial draw."""
    p0 = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)
    out = list(params)
    out[2] = p0[2]
    if SP.recon(cfg):
        out[4] = p0[4]
    return tuple(out)


def attach_init_heads(params, cfg, Xtr):
    """A head-free (case none) encoder with cfg's heads at their initial draw; mu (and xbar)
    measured on THIS encoder's stream."""
    p0 = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)
    return SP.init_mu((params[0], params[1], *p0[2:]), SP.mu_rows(Xtr, cfg), cfg)


@eqx.filter_jit
def readout(params, x, cfg, t):
    """-> (embedding, activities per layer, codes per head)."""
    if cfg.sparse.case == "D":
        acts, ex = SP.deploy(params, x, cfg, t=t)
        return acts[-1], tuple(acts), tuple(ex["a"])
    z, hs = E.forward(params, x, cfg.enc)
    return z, tuple([*hs, z]), ()


def _feats(params, cfg, X, t):
    outs = [readout(params, X[i:i + 1000], cfg, t) for i in range(0, len(X), 1000)]
    z = np.concatenate([np.asarray(o[0]) for o in outs])
    acts = [np.concatenate([np.asarray(o[1][l]) for o in outs]) for l in range(len(outs[0][1]))]
    codes = [np.concatenate([np.asarray(o[2][h]) for o in outs]) for h in range(len(outs[0][2]))]
    return z, acts, codes


def analyse_one(params, cfg, role, tag, data, rows, emb=True, subsets=None):
    """One (params, cfg) under one role.  subsets: concatenation sets to probe
    besides 'all' (the floor gets every nested prefix so K<5 runs have a match)."""
    xtr, ytr_, xte, yte_ = data
    sp = cfg.sparse
    K = len(sp.heads) if (sp.case == "D" and role == "trained_D") else 0

    def add(metric, value, condition="clean", t="", head=""):
        rows.append({"run": tag, "role": role, "case": sp.case, "arm": cfg.arm, "seed": cfg.opt.seed,
                     "K": K, "heads_attached": len(sp.heads) if sp.case == "D" else 0,
                     "metric": metric, "condition": condition, "t": t, "head": head, "value": value})

    D_on = sp.case == "D"
    ts = [("0", 0), ("T", sp.T_test)] if D_on else [("", None)]
    conds = [("clean", 0.0)] + [("noise", s) for s in NOISE] + [("occlusion", o) for o in OCCL]
    ftr = {lab: _feats(params, cfg, xtr, t) for lab, t in ts}
    tr_T = ftr[ts[-1][0]]
    if emb:
        te = _feats(params, cfg, xte, ts[-1][1])
        for l in range(len(te[1])):
            add("input_spread", _spread(te[1][l]), head=str(l))
            add("layer_norm", float(np.linalg.norm(te[1][l], axis=1).mean()), head=str(l))
        add("embed_spread", _spread(te[0]))
    groups = {}
    if D_on:
        hs = list(sp.heads)
        groups["all"] = hs
        for sub in (subsets or []):
            groups["+".join(map(str, sub))] = list(sub)
    for kind, lvl in conds:
        xc = corrupt(xte, cfg, kind, lvl)
        cname = kind if kind == "clean" else f"{kind}={lvl:g}"
        got = {}
        for lab, t in ts:
            z_te, a_te, c_te = _feats(params, cfg, xc, t)
            z_tr, a_tr, c_tr = ftr[lab]
            if emb and lab == ts[-1][0]:             # embedding and f_l: t-independent, written once
                for k, v in probe_pair(z_tr, ytr_, z_te, yte_).items():
                    add(f"embed_{k}", v, cname)
                for l in (sp.heads if D_on else []):
                    for k, v in probe_pair(a_tr[l], ytr_, a_te[l], yte_).items():
                        add(f"f_{k}", v, cname, head=str(l))
                if D_on:
                    for k, v in probe_pair(np.concatenate([a_tr[l] for l in hs], 1), ytr_,
                                           np.concatenate([a_te[l] for l in hs], 1), yte_).items():
                        add(f"f_{k}", v, cname, head="all")
            if not D_on:
                continue
            for h, l in enumerate(sp.heads):
                pr = probe_pair(c_tr[h], ytr_, c_te[h], yte_, sd_floor=CODE_SD_FLOOR)
                got[(lab, str(l))] = pr
                for k, v in pr.items():
                    add(f"code_{k}", v, cname, lab, str(l))
                if kind == "clean":
                    on = c_te[h] != 0
                    add("code_active", float(on.mean()), cname, lab, str(l))
                    add("code_supports", len({r.tobytes() for r in on}), cname, lab, str(l))
                    add("code_dead_atoms", float(np.mean(~on.any(0))), cname, lab, str(l))
            for gname, gl in groups.items():
                idx = [sp.heads.index(l) for l in gl]
                pr = probe_pair(np.concatenate([c_tr[i] for i in idx], 1), ytr_,
                                np.concatenate([c_te[i] for i in idx], 1), yte_, sd_floor=CODE_SD_FLOOR)
                got[(lab, gname)] = pr
                for k, v in pr.items():
                    add(f"code_{k}", v, cname, lab, gname)
        if D_on:
            for h in [str(l) for l in sp.heads] + list(groups):
                for k in ("linear", "knn"):
                    add(f"delta_inf_code_{k}", got[("T", h)][k] - got[("0", h)][k], cname, "T-0", h)
    return rows


def analyse(run_dir, n_train, n_test, heads_json=None):
    cfg, params = load_run(run_dir)
    tag = os.path.basename(run_dir.rstrip("/"))
    Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
    rtr = np.random.default_rng(0).choice(len(Xtr), min(n_train, len(Xtr)), replace=False)
    rte = np.random.default_rng(1).choice(len(Xte), min(n_test, len(Xte)), replace=False)
    data = (D.eval_batch(jnp.asarray(Xtr[rtr]), cfg.data), np.asarray(ytr)[rtr],
            D.eval_batch(jnp.asarray(Xte[rte]), cfg.data), np.asarray(yte)[rte])
    rows = []
    nested = [tuple(sorted(ORDER[:k])) for k in range(1, len(ORDER))]
    if cfg.sparse.case == "D":
        floor = "randD" in tag
        role = "floor_init" if floor else "trained_D"
        analyse_one(params, cfg, role, tag, data, rows, subsets=nested if floor else None)
        if not floor:      # same trained encoder, dictionaries put back to their initial draw
            # trained encoder, trained mu (it describes the encoder's stream), initial D
            analyse_one(with_init_dicts(params, cfg), cfg, "trained_enc_init_D", tag, data, rows, emb=False)
    else:
        analyse_one(params, cfg, "reused_headfree", tag, data, rows)
        if heads_json:
            c2 = dataclasses.replace(cfg, sparse=k5_sparse(heads_json))
            pr = attach_init_heads(params, c2, Xtr)
            analyse_one(pr, c2, "reused_enc_init_D", tag, data, rows,
                        emb=False, subsets=nested)
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
            if r["condition"] in ("clean", "noise=0.2") and r["metric"] in ("embed_linear", "code_linear", "f_linear") \
                    and r["head"] in ("", "all") and r["t"] in ("", "T"):
                print(f"   {r['role']:<20} {r['metric']:<13} {r['condition']:<10} {r['value']:.4f}", flush=True)
        with open(args.csv, "w", newline="") as f:          # rewritten after every run
            w = csv.DictWriter(f, fieldnames=list(allr[0].keys()))
            w.writeheader(); w.writerows(allr)
    print(f"wrote {args.csv} ({len(allr)} rows)", flush=True)


if __name__ == "__main__":
    main()
