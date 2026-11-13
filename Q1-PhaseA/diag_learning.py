# CLI: for each trained run, measures per-layer weight displacement from its re-seeded init and the kNN/grad-norm
# trajectory from log.json, flags layers frozen near the displacement floor, and writes the results to a CSV.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equinox as eqx
import jax
import numpy as np

import encoder as E
import sparse as SP
from run_probes import cfg_from_dict


def _arrays(tree):
    """Converts all inexact-array leaves of a pytree into a flat list of numpy arrays."""
    return [np.asarray(x) for x in
            jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))]


def weight_displacement(run_dir):
    """Computes relative weight displacement between initial and trained encoder layers (excludes case-C dict).
    Returns per-layer ratios, overall relative norm, and layer shapes; stores dict displacement separately."""
    cfg = cfg_from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    init = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)
    trained = eqx.tree_deserialise_leaves(os.path.join(run_dir, "params.eqx"), init)
    # encoder layers only: a case-C dictionary (params[2]) is NOT a layer -- it
    # is re-normalised every step, so its displacement has another scale and
    # would set the floor / shift n_frozen (found in review).  -> dict_disp.
    A, B = _arrays((init[0], init[1])), _arrays((trained[0], trained[1]))
    per = [float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))
           for a, b in zip(A, B)]
    num = float(np.sqrt(sum(((b - a) ** 2).sum() for a, b in zip(A, B))))
    den = float(np.sqrt(sum((a ** 2).sum() for a in A))) + 1e-12
    if len(init) > 2:
        d0, d1 = np.asarray(init[2]), np.asarray(trained[2])
        weight_displacement.dict_disp = float(np.linalg.norm(d1 - d0) / np.linalg.norm(d0))
    else:
        weight_displacement.dict_disp = None
    return per, num / den, [a.shape for a in A]


def probe_trajectory(run_dir):
    """Builds a dict of first/best/last kNN accuracy and grad-norm stats from a run's log.json.
    knn_best_minus_first <= 0 means the encoder never beat its initial embedding."""
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
    """Builds a summary row of weight displacement and probe trajectory stats for a run, optionally printing it.
    Counts layers "frozen" as within 1.5x of the minimum displacement, since decay alone yields a small uniform shift."""
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
    if weight_displacement.dict_disp is not None:
        row["dict_disp"] = weight_displacement.dict_disp
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
    """Parses run directories from the command line and reports diagnostics for each one with a params.eqx file.
    Writes the collected results to a CSV file specified by --csv."""
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
