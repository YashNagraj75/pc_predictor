"""Is the PC inference step size right?  Batch-invariance + stability sweep.

    ./.venv/bin/python diag_inference.py --run runs/pc_sigreg_T16_coupled_e400

WHY THIS EXISTS
---------------
`jpc.pc_energy_fn` returns a MEAN over the batch, so the energy gradient with
respect to one sample's activity is 1/B of that sample's own gradient.  An
`activity_lr` applied directly to that gradient therefore does not mean what it
looks like: the realised per-sample step is `activity_lr / B`, and it shrinks as
the batch grows.  At batch_size=256 our nominal 0.5 was a realised 0.002, about
two orders of magnitude under the stable optimum, and the relaxation barely
moved off the feedforward initialisation.

That is the whole depth-starvation signature: with too small a step, the credit
wave launched at the output decays before it reaches the deep layers, so the
deep weights see no gradient and only weight decay moves them.  It looks exactly
like a fundamental limitation of predictive coding and is in fact a step-size
bug.

Two checks, both cheap, run on ONE batch at initialisation:

1. Batch invariance.  One sample's settled activity must not depend on how many
   other samples share its batch.  Fails iff the 1/B factor is unhandled.  This
   mirrors the regression test in SakanaAI/pc-alm, which documents the same
   pitfall and fixes it with `effective_lr = state_lr * batch_size`.

2. Stability sweep.  Sweep the step and report, per step size, the final energy,
   the final objective, whether the objective fell monotonically, and how many
   layers actually moved.  The usable step is the largest one that is still
   monotone; above it the iteration overshoots and then diverges.

Run this whenever depth, width, batch size, or the objective changes -- the
stable step depends on sigma_max of the constraint operator and therefore on all
four.
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
import sigreg as S
import train as T
from run_probes import cfg_from_dict


def _feedforward_layers(params, zin, enc):
    """Forward activities in the same [h_1..h_{L-1}, z_out] order as pc_relax."""
    z, hs = E.forward(params, zin, enc)
    return [*hs, z]


def _objective(params, Zin, A, cfg):
    Zv = jnp.stack([E.forward(params, Zin[v], cfg.enc)[0] for v in range(Zin.shape[0])])
    return float(S.lejepa_loss(Zv, A, cfg.data.n_global_views, cfg.sig.lambd,
                               cfg.use_sigreg, projector=cfg.projector,
                               t_max=cfg.sig.t_max, n_points=cfg.sig.n_points,
                               scale_by_n=cfg.sig.scale_by_n)[0])


def batch_invariance(params, Xtr, A, cfg, key, sizes=(2, 8, 32)):
    """-> list of (B, relative movement of sample 0's embedding).

    Constant across B  => the per-sample step is batch-invariant (correct).
    Falling with B      => the 1/B mean factor is unhandled (bug).
    """
    out = []
    for B in sizes:
        Zin, _ = D.make_views(key, jnp.asarray(Xtr[:B]), cfg.data)
        acts, _ = T.pc_relax(params, Zin, A, cfg, key)
        z0 = E.forward(params, Zin[0], cfg.enc)[0][0]
        moved = float(jnp.linalg.norm(acts[0][-1][0] - z0) / (jnp.linalg.norm(z0) + 1e-12))
        out.append((B, moved))
    return out


def stability_sweep(params, Xtr, A, cfg, key, lrs, batch=64):
    """-> list of row dicts, one per candidate eta_h (PER-SAMPLE step)."""
    Zin, _ = D.make_views(key, jnp.asarray(Xtr[:batch]), cfg.data)
    ffl = _feedforward_layers(params, Zin[0], cfg.enc)
    rows = []
    for lr in lrs:
        cfg2 = dataclasses.replace(cfg, pc=dataclasses.replace(cfg.pc, eta_h=lr))
        acts, hist = T.pc_relax(params, Zin, A, cfg2, key)
        Fh, Lh = np.asarray(hist["F"]), np.asarray(hist["L"])
        per = [float(jnp.linalg.norm(acts[0][i] - ffl[i]) / (jnp.linalg.norm(ffl[i]) + 1e-12))
               for i in range(len(ffl))]
        diverged = (not np.isfinite(Fh[-1])) or Fh[-1] > 1e4
        rows.append({
            "eta_h": lr,
            "F_final": float(Fh[-1]),
            "L_final": float(Lh[-1]),
            # monotone objective is the practical stability test: the relaxation
            # is supposed to lower the objective at every inference step.
            "L_monotone": bool(np.all(np.diff(Lh) <= 1e-6)),
            "n_layers_moved": int(sum(1 for v in per if v > 1e-3)),
            "n_layers": len(per),
            "diverged": bool(diverged),
            **{f"L{i}_move": v for i, v in enumerate(per)},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/pc_sigreg_T16_coupled_e400",
                    help="run dir to read config.json from")
    ap.add_argument("--batch", type=int, default=256,
                    help="sweep at the batch size you will TRAIN at")
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[0.0156, 0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375])
    ap.add_argument("--csv", default="diag_inference.csv")
    args = ap.parse_args()

    cfg = cfg_from_dict(json.load(open(os.path.join(args.run, "config.json"))))
    params = E.init(jax.random.PRNGKey(cfg.opt.seed), cfg.enc, cfg.data.input_dim)
    Xtr = D.load_dataset(cfg.data)[0]
    kv, ka, _ = jax.random.split(jax.random.PRNGKey(0), 3)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)

    print(f"config: eta_h={cfg.pc.eta_h}  T={cfg.pc.T}  "
          f"train batch_size={cfg.opt.batch_size}  output_drive={cfg.pc.output_drive}")

    print("\n-- batch invariance (movement of sample 0; should be CONSTANT in B) --")
    inv = batch_invariance(params, Xtr, A, cfg, kv)
    for B, moved in inv:
        print(f"   B={B:3d}  rel movement = {moved:.6e}")
    ratio = inv[0][1] / max(inv[-1][1], 1e-30)
    print(f"   ratio B={inv[0][0]} : B={inv[-1][0]}  =  {ratio:.1f}x   "
          f"({'BATCH-DEPENDENT -- step needs a *B correction' if ratio > 1.5 else 'ok'})")

    print(f"\n-- stability sweep at B={args.batch}, T={cfg.pc.T} --")
    L_ff = _objective(params, D.make_views(kv, jnp.asarray(Xtr[:args.batch]), cfg.data)[0],
                      A, cfg)
    print(f"   objective at the feedforward embedding = {L_ff:.6f}")
    rows = stability_sweep(params, Xtr, A, cfg, kv, args.lrs, batch=args.batch)
    print(f"   {'eta_h':>8} {'per-sample':>11} {'F_final':>11} {'L_final':>10} "
          f"{'monotone':>9} {'moved':>7}")
    for r in rows:
        flag = "  DIVERGED" if r["diverged"] else ""
        print(f"   {r['eta_h']:8.3g} {r['eta_h']:11.4g} "
              f"{r['F_final']:11.4g} {r['L_final']:10.4g} "
              f"{str(r['L_monotone']):>9} {r['n_layers_moved']:>4}/{r['n_layers']}{flag}")

    # SELECTION RULE.  Not "the largest stable step" -- that picks the edge of
    # stability, and at B=256 the edge is measurably worse than a smaller step
    # (eta_h=0.25 lands at objective 0.644 while 0.0625 reaches 0.557).  What
    # we actually want is the relaxation that gets FURTHEST DOWN the objective
    # within the T budget, among the steps that descend cleanly.  Monotonicity
    # stays as a filter because a step that oscillates leaves erratic residuals,
    # and the residuals are what the weight update is built from.
    usable = [r for r in rows if r["L_monotone"] and not r["diverged"]]
    if not usable:
        usable = [r for r in rows if not r["diverged"]]
        print("\n   WARNING: no step descended monotonically; "
              "falling back to non-diverged steps only")
    if usable:
        best = min(usable, key=lambda r: r["L_final"])
        print(f"\n   SELECTED eta_h={best['eta_h']:g}  -- best objective among "
              f"{len(usable)} clean steps: {L_ff:.4f} -> {best['L_final']:.4f}, "
              f"{best['n_layers_moved']}/{best['n_layers']} layers moving")
        widest = max(usable, key=lambda r: (r["n_layers_moved"], -r["L_final"]))
        if widest["eta_h"] != best["eta_h"]:
            print(f"   (note: eta_h={widest['eta_h']:g} moves more layers, "
                  f"{widest['n_layers_moved']}/{widest['n_layers']}, but only "
                  f"reaches {widest['L_final']:.4f})")
        with open("eta_h_selected.txt", "w") as f:
            f.write(f"{best['eta_h']:g}\n")

    keys = sorted({k for r in rows for k in r})
    keys = ["eta_h"] + [k for k in keys if k != "eta_h"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv} ({len(rows)} step sizes)")


if __name__ == "__main__":
    main()
