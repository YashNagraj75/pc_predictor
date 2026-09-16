"""Did the encoder LEARN?  Per-layer weight displacement + probe trajectory.

    ./.venv/bin/python diag_learning.py runs/bp_sigreg_e400 runs/pc_sigreg_T16_coupled_e400
    ./.venv/bin/python diag_learning.py runs/* --csv diag_learning.csv

WHY THIS EXISTS
---------------
probes.py answers "what does the final embedding contain".  It cannot answer
"did training do anything", and those came apart badly here: the PC arm's
probe scores sat inside the range of an UNTRAINED encoder, so every
representation number it produced was describing the random initialisation
rather than anything the learning rule had built.

Two measurements, both cheap, both read off artifacts that already exist:

1. Per-layer relative weight displacement ||W_end - W_init|| / ||W_init||,
   recomputing W_init from the run's own seed.  This localises the learning
   signal by DEPTH.  A rule whose updates reach only the last few layers
   leaves the deep stack -- which is what builds the representation the probe
   reads -- at initialisation, and no amount of probing the embedding will
   tell you that is what happened.

2. The kNN trajectory already logged during training, reported as
   best-minus-first rather than last-minus-first.  If the best value over the
   whole run is the value at step 0, training never improved the
   representation at any point, which is a categorically different statement
   from "it ended lower" and rules out "needs a longer budget".

Run this BEFORE interpreting any probe table from a new arm.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equinox as eqx
import jax
import numpy as np

import encoder as E
from run_probes import cfg_from_dict


def _arrays(tree):
    return [np.asarray(x) for x in
            jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))]


def weight_displacement(run_dir):
    """-> (per_layer list, overall float, shapes list).

    W_init is REBUILT from the run's recorded seed rather than stored, so this
    works on any run that saved params.eqx + config.json -- including runs that
    finished long before this diagnostic existed.
    """
    cfg = cfg_from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    init = E.init(jax.random.PRNGKey(cfg.opt.seed), cfg.enc, cfg.data.input_dim)
    trained = eqx.tree_deserialise_leaves(os.path.join(run_dir, "params.eqx"), init)
    A, B = _arrays(init), _arrays(trained)
    per = [float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))
           for a, b in zip(A, B)]
    num = float(np.sqrt(sum(((b - a) ** 2).sum() for a, b in zip(A, B))))
    den = float(np.sqrt(sum((a ** 2).sum() for a in A))) + 1e-12
    return per, num / den, [a.shape for a in A]


def probe_trajectory(run_dir):
    """-> dict from the training log: first/best/last kNN and grad-norm scale.

    `best_minus_first` is the number that matters.  <= 0 means the encoder was
    never better than its initialisation at any evaluated step.
    """
    p = os.path.join(run_dir, "log.json")
    if not os.path.exists(p):
        return {}
    log = json.load(open(p))
    ev = [(r["step"], r["eval"]["embedding"]["knn"]) for r in log
          if isinstance(r.get("eval"), dict)
          and "knn" in r.get("eval", {}).get("embedding", {})]
    gn = np.array([r["grad_norm_preclip"] for r in log
                   if "grad_norm_preclip" in r], dtype=float)
    out = {}
    if ev:
        k = [v for _, v in ev]
        out.update(knn_first=k[0], knn_best=max(k), knn_last=k[-1],
                   knn_best_step=ev[int(np.argmax(k))][0],
                   knn_best_minus_first=max(k) - k[0], n_eval=len(ev))
    if gn.size:
        out.update(grad_norm_median=float(np.median(gn)),
                   grad_norm_first100=float(gn[:100].mean()),
                   grad_norm_last100=float(gn[-100:].mean()))
    return out


def report(run_dir, verbose=True):
    per, overall, shapes = weight_displacement(run_dir)
    traj = probe_trajectory(run_dir)
    tag = os.path.basename(os.path.normpath(run_dir))
    # "frozen" = within 1.5x of the smallest displacement in this run: with
    # zero gradient, weight decay alone still produces a small, near-identical
    # relative shift in every starved layer, so the floor is the tell.
    floor = min(per)
    n_frozen = sum(1 for v in per if v <= 1.5 * floor)
    row = {"tag": tag, "overall_disp": overall, "disp_floor": floor,
           "n_layers": len(per), "n_frozen": n_frozen, **traj}
    row.update({f"L{i}_disp": v for i, v in enumerate(per)})
    if verbose:
        print(f"\n=== {tag} ===")
        print(f"  overall displacement {overall:.5f}   "
              f"layers at the floor ({floor:.5f}): {n_frozen}/{len(per)}")
        print("  per-layer: " + " ".join(f"{v:.4f}" for v in per))
        if traj:
            print(f"  knn first={traj.get('knn_first'):.4f} "
                  f"best={traj.get('knn_best'):.4f} "
                  f"(step {traj.get('knn_best_step')}) "
                  f"last={traj.get('knn_last'):.4f}  "
                  f"BEST-FIRST={traj.get('knn_best_minus_first'):+.4f}")
            print(f"  grad norm median={traj.get('grad_norm_median'):.3e}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", default="diag_learning.csv")
    args = ap.parse_args()

    rows = [report(d) for d in args.runs
            if os.path.exists(os.path.join(d, "params.eqx"))]
    if rows:
        import csv
        keys = sorted({k for r in rows for k in r})
        keys = ["tag"] + [k for k in keys if k != "tag"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
