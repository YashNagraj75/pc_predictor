"""Pick alpha (and re-check eta_h) for PC-ALM by gradient agreement with backprop.

    ./.venv/bin/python diag_alm.py --batch 256
    ./.venv/bin/python diag_alm.py --batch 256 --run runs/pc_sigreg_T16_coupled_etafix_s0

WHY NOT THE diag_inference.py CRITERIA
--------------------------------------
diag_inference.py selects eta_h by "how far down the objective does the
relaxation get, while descending cleanly", and reports how many layers moved
off the feedforward initialisation.  Both are sensible for PLAIN PC, where the
relaxation's whole job is to drag the activities toward lower loss and the
weight update then chases the resulting residuals.

They are the wrong instrument for PC-ALM, and would select against it.
PC-ALM's fixed point is PRIMAL FEASIBLE: r -> 0, the activities return to the
feedforward pass, and the credit is carried by the duals instead.  So as alpha
does its job, activity displacement goes DOWN and the relaxed objective goes
back UP toward the feedforward value.  Ranking alpha by "lowest relaxed
objective" or "most layers moved" would therefore prefer alpha = 0.

What PC-ALM actually claims is that its WEIGHT GRADIENT becomes the backprop
weight gradient -- the duals integrate to the adjoint at every layer.  That is
a directly measurable quantity at initialisation, needs no training, and is
what this file ranks on:

    cos( PC-ALM weight gradient , backprop weight gradient )

Three things it reports:

1. cos vs alpha at our inference budget, for each candidate eta_h.  The
   selection.  alpha=0 must reproduce the plain-PC cosine -- that is the
   sanity check that this script and the training path agree.
2. cos vs T at the selected alpha.  The paper's Figure 5 shape: PC's agreement
   should grow like sqrt(T) (diffusive) and PC-ALM's like T (ballistic).
3. Per-layer cosine at the selected setting.  The depth question restated
   correctly: BP agreement should hold at layer 1, not just near the output.
   A global cosine can look healthy while the input layers are uncorrelated.

Divergence guard: any setting whose gradient is non-finite, or whose norm
exceeds the backprop norm by more than --max-norm-ratio, is excluded before
ranking.  A large mis-scaled gradient can still have a high cosine.
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
import sigreg as S
import train as T
from config import OptCfg, PCCfg, RunCfg
from run_probes import cfg_from_dict


def _leaves(g):
    return [x for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
            if jnp.issubdtype(x.dtype, jnp.floating)]


def cosine(ga, gb):
    a, b = _leaves(ga), _leaves(gb)
    dot = sum(jnp.sum(x * y) for x, y in zip(a, b))
    na = jnp.sqrt(sum(jnp.sum(x ** 2) for x in a))
    nb = jnp.sqrt(sum(jnp.sum(y ** 2) for y in b))
    return float(dot / (na * nb + 1e-30)), float(na), float(nb)


def per_layer_cosine(ga, gb):
    out = []
    for x, y in zip(_leaves(ga), _leaves(gb)):
        d = float(jnp.sum(x * y))
        n = float(jnp.linalg.norm(x) * jnp.linalg.norm(y))
        out.append(d / (n + 1e-30))
    return out


def make_batch(cfg, Xtr, seed=0):
    kv, ka = jax.random.split(jax.random.PRNGKey(seed))
    kb = jax.random.PRNGKey(seed + 99)
    idx = jax.random.choice(kb, Xtr.shape[0], (cfg.opt.batch_size,), replace=False)
    Zin, _ = D.make_views(kv, Xtr[idx], cfg.data)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
    return Zin, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="optional run dir to load TRAINED weights from; "
                         "default is a fresh initialisation")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0625, 0.125])
    ap.add_argument("--Ts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--max-norm-ratio", type=float, default=50.0)
    ap.add_argument("--csv", default="diag_alm.csv")
    args = ap.parse_args()

    if args.run:
        cfg = cfg_from_dict(json.load(open(os.path.join(args.run, "config.json"))))
        skeleton = E.init(jax.random.PRNGKey(cfg.opt.seed), cfg.enc, cfg.data.input_dim)
        params = eqx.tree_deserialise_leaves(os.path.join(args.run, "params.eqx"), skeleton)
        where = f"trained weights from {args.run}"
    else:
        cfg = RunCfg(arm="pc", use_sigreg=True, pc=PCCfg(T=args.T),
                     opt=OptCfg(seed=0, batch_size=args.batch))
        params = E.init(jax.random.PRNGKey(cfg.opt.seed), cfg.enc, cfg.data.input_dim)
        where = "fresh initialisation"
    cfg = dataclasses.replace(cfg, opt=dataclasses.replace(cfg.opt, batch_size=args.batch))

    Xtr = jnp.asarray(D.load_dataset(cfg.data)[0])
    Zin, A = make_batch(cfg, Xtr)
    print(f"target: backprop gradient on one batch of {args.batch}, {where}")

    g_bp, _ = T.bp_grads(params, Zin, A, cfg)
    _, n_bp, _ = cosine(g_bp, g_bp)
    print(f"        ||g_bp|| = {n_bp:.4e}\n")

    # SELECTION METRIC.  Not the global cosine: that is a magnitude-weighted
    # inner product, so a few high-norm layers dominate it and a setting can
    # score 0.50 globally while nine of ten layers agree with backprop above
    # 0.94.  Ranking on it would rank by gradient scale, not by how deep
    # BP-equivalent credit reaches -- which is the entire question here.
    # Rank on the MEAN PER-LAYER cosine (equal weight per layer, so depth
    # counts as much as magnitude), and report the agreement front -- how many
    # layers clear 0.9 -- alongside it.
    rows = []
    hdr = (f"{'eta_h':>8} {'alpha':>7} {'mean/layer':>11} {'front>0.9':>10} "
           f"{'global cos':>11} {'|g|/|g_bp|':>11} {'ok':>4}")
    print(hdr)
    for eta in args.etas:
        for al in args.alphas:
            c2 = dataclasses.replace(cfg, pc=dataclasses.replace(
                cfg.pc, T=args.T, eta_h=eta, alpha=al))
            g, aux = T.pc_grads(params, Zin, A, c2, jax.random.PRNGKey(0))
            cos, na, nb = cosine(g, g_bp)
            pl = per_layer_cosine(g, g_bp)
            mean_pl = float(np.mean(pl))
            front = int(sum(1 for v in pl if v > 0.9))
            ratio = na / (nb + 1e-30)
            ok = (np.isfinite(cos) and np.isfinite(ratio) and np.all(np.isfinite(pl))
                  and ratio <= args.max_norm_ratio)
            rows.append({"eta_h": eta, "alpha": al, "T": args.T, "cos": cos,
                         "mean_layer_cos": mean_pl, "front_0p9": front,
                         "norm_ratio": ratio, "dual_norm": float(aux["dual_norm"]),
                         "F_final": float(aux["F_final"]), "L_final": float(aux["L_final"]),
                         "ok": bool(ok),
                         **{f"L{i}_cos": v for i, v in enumerate(pl)}})
            print(f"{eta:8.4g} {al:7.3g} {mean_pl:11.5f} {front:>7d}/{len(pl)} "
                  f"{cos:11.5f} {ratio:11.4g} {'y' if ok else 'NO':>4}")

    usable = [r for r in rows if r["ok"]]
    if not usable:
        print("\nNO USABLE SETTING -- every candidate diverged or blew up in norm")
        sys.exit(2)
    best = max(usable, key=lambda r: (r["mean_layer_cos"], r["front_0p9"]))
    base = [r for r in usable if r["alpha"] == 0.0 and r["eta_h"] == best["eta_h"]]
    print(f"\nSELECTED eta_h={best['eta_h']:g} alpha={best['alpha']:g}  "
          f"mean/layer cos={best['mean_layer_cos']:.5f}  "
          f"front={best['front_0p9']}/{len(_leaves(g_bp))}  global={best['cos']:.5f}")
    if base:
        b0 = base[0]
        print(f"   plain PC at the same eta_h: mean/layer {b0['mean_layer_cos']:.5f} "
              f"front {b0['front_0p9']}  global {b0['cos']:.5f}   "
              f"(mean/layer improvement {best['mean_layer_cos']-b0['mean_layer_cos']:+.5f})")
        if best["alpha"] == 0.0:
            print("   NOTE: alpha=0 won -- the dual term did not improve BP agreement here.")
    with open("alm_selected.txt", "w") as f:
        f.write(f"{best['eta_h']:g} {best['alpha']:g}\n")

    # --- cos vs T: diffusive (sqrt T) vs ballistic (T) ---------------------
    print(f"\ncos vs inference budget at eta_h={best['eta_h']:g}  "
          f"(alpha=0 vs alpha={best['alpha']:g})")
    print(f"{'T':>5} {'alpha=0':>12} {'alpha=sel':>12}")
    for Tt in args.Ts:
        line = [Tt]
        for al in (0.0, best["alpha"]):
            c2 = dataclasses.replace(cfg, pc=dataclasses.replace(
                cfg.pc, T=Tt, eta_h=best["eta_h"], alpha=al))
            g, _ = T.pc_grads(params, Zin, A, c2, jax.random.PRNGKey(0))
            line.append(cosine(g, g_bp)[0])
        print(f"{line[0]:5d} {line[1]:12.5f} {line[2]:12.5f}")
        rows.append({"eta_h": best["eta_h"], "alpha": None, "T": Tt,
                     "cos_alpha0": line[1], "cos_alpha_sel": line[2], "ok": True})

    # --- per-layer cosine: is the agreement uniform in depth? --------------
    print(f"\nper-layer cos(g_alm, g_bp) at the selected setting, and at alpha=0")
    for al in (0.0, best["alpha"]):
        c2 = dataclasses.replace(cfg, pc=dataclasses.replace(
            cfg.pc, T=args.T, eta_h=best["eta_h"], alpha=al))
        g, _ = T.pc_grads(params, Zin, A, c2, jax.random.PRNGKey(0))
        pl = per_layer_cosine(g, g_bp)
        print(f"   alpha={al:<5g} " + " ".join(f"{v:6.3f}" for v in pl))
        rows.append({"eta_h": best["eta_h"], "alpha": al, "T": args.T,
                     **{f"L{i}_cos": v for i, v in enumerate(pl)}, "ok": True})

    keys = sorted({k for r in rows for k in r})
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
