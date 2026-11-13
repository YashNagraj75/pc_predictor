# CLI: loads each saved run's params and config (cfg_from_dict), encodes train/test images (plain and augmented),
# and runs the full probe suite (classification, regression, decode, kNN) on the resulting embeddings, writing
# probes.json per run and an aggregate probes_summary.csv. Also accepts "random:<seed>" for an untrained-encoder floor.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equinox as eqx
import jax
import numpy as np

import data as D
import encoder as E
import probes as P
import sparse as SP
from config import DataCfg, EncoderCfg, OptCfg, PCCfg, RunCfg, SIGRegCfg, SparseCfg


def cfg_from_dict(d):
    """Rebuilds a RunCfg from a saved config dict, reconstructing nested dataclasses and list-to-tuple fields.
    Falls back to an empty dict for "sparse" since older runs predate SparseCfg."""
    def mk(kls, sub):
        """Builds a `kls` instance from `sub`, keeping only keys matching its dataclass fields.
        Converts list values to tuples before passing them as keyword arguments."""
        fields = {f.name: f.type for f in kls.__dataclass_fields__.values()}
        kw = {}
        for k, v in sub.items():
            if k not in fields:
                continue
            kw[k] = tuple(v) if isinstance(v, list) else v
        return kls(**kw)
    top = {k: v for k, v in d.items()
           if k not in ("data", "enc", "sig", "pc", "opt", "sparse")}
    # runs written before SparseCfg existed have no "sparse" block -> case none
    return RunCfg(**top, data=mk(DataCfg, d["data"]), enc=mk(EncoderCfg, d["enc"]),
                  sig=mk(SIGRegCfg, d["sig"]), pc=mk(PCCfg, d["pc"]),
                  opt=mk(OptCfg, d["opt"]), sparse=mk(SparseCfg, d.get("sparse", {})))


def load_run(run_dir):
    """Loads the config and reconstructs model parameters from a saved run directory.
    Returns the config object and the deserialized equinox params."""
    cfg = cfg_from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    skeleton = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, cfg.data.input_dim)
    params = eqx.tree_deserialise_leaves(os.path.join(run_dir, "params.eqx"),
                                         skeleton)
    return cfg, params


def embed_split(params, cfg, X, n, seed=0):
    """Computes deterministic latent codes Z for a random subset of X via the deployed network.
    Returns Z alongside the resized/grayscaled images actually fed to the encoder, plus the sampled indices."""
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    xb = D.eval_batch(jax.numpy.asarray(X[idx]), cfg.data)
    z, _ = SP.forward(params, xb, cfg)      # deployed net; == E.forward for case none
    c = 1 if cfg.data.grayscale else 3
    imgs = np.asarray(xb).reshape(-1, cfg.data.img_size, cfg.data.img_size, c)
    return np.asarray(z), imgs, idx


def embed_augmented(params, cfg, X, n, seed=0):
    """Embeds one augmented view per sampled image and returns its latent codes, boxes, and aug params.
    Keeps only the first view per image so rows stay independent and avoid train/test leakage."""
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    key = jax.random.PRNGKey(10_000 + seed)
    views, boxes, prms = D.make_views_labeled(
        key, jax.numpy.asarray(X[idx]), cfg.data)
    z, _ = SP.forward(params, views[0], cfg)
    return np.asarray(z), np.asarray(boxes[0]), np.asarray(prms[0])


def main():
    """Runs probe suites on each checkpoint (or an untrained-encoder floor) and writes probes.json plus probes_summary.csv.
    Includes a kNN probe for comparability with historical numbers; random floors use the ORIGINAL architecture only."""
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n-train", type=int, default=6000,
                    help="probe/decoder fitting set size")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--decoder-steps", type=int, default=3000)
    ap.add_argument("--fast", action="store_true",
                    help="skip MLP probes and shorten the decoder -- smoke only")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    rows = []
    for run_dir in args.runs:
        # The random:<seed> reference has no params.eqx by definition, so this
        # branch must come BEFORE the missing-weights guard below.
        if not run_dir.startswith("random:") and \
                not os.path.exists(os.path.join(run_dir, "params.eqx")):
            print(f"[skip] {run_dir}: no params.eqx", flush=True)
            continue
        if run_dir.startswith("random:"):
            # Untrained-encoder reference: not a nicety, since PC's trained kNN probe once landed BELOW this floor, so
            # "beats backprop" and "beats doing nothing" are different questions. Computed under the same probes and
            # split as every trained arm -- the prior 0.343 reference was kNN-only and the two probes disagree.
            sd = int(run_dir.split(":", 1)[1])
            cfg = RunCfg(opt=OptCfg(seed=sd))
            # ORIGINAL architecture only.  Sparse arms (B/C) need their own floor
            # from `run_arms.py --init-only` (skip at s removed, W_s rescaled, D).
            print(f"[floor] random:{sd} = untrained ORIGINAL-architecture encoder; "
                  f"not a floor for sparse arms", flush=True)
            params = E.init(jax.random.PRNGKey(sd), cfg.enc, cfg.data.input_dim)
            run_dir = os.path.join("runs", f"random_init_seed{sd}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            cfg, params = load_run(run_dir)
        Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
        Ztr, Itr, itr = embed_split(params, cfg, Xtr, args.n_train, seed=0)
        Zte, Ite, ite = embed_split(params, cfg, Xte, args.n_test, seed=1)

        Zatr, Batr, Patr = embed_augmented(params, cfg, Xtr, args.n_train, seed=0)
        Zate, Bate, Pate = embed_augmented(params, cfg, Xte, args.n_test, seed=1)

        tag = os.path.basename(os.path.normpath(run_dir))
        print(f"\n=== {tag}  Z{Ztr.shape} -> img{Itr.shape[1:]} ===", flush=True)
        rep = P.run_suite(
            Ztr, Zte, Itr, Ite, np.asarray(ytr)[itr], np.asarray(yte)[ite],
            out_dir=run_dir, tag="",
            boxes_tr=Batr, boxes_te=Bate,
            aug_params_tr=Patr, aug_params_te=Pate,
            Zaug_tr=Zatr, Zaug_te=Zate,
            decoder_steps=400 if args.fast else args.decoder_steps,
            probe_mlp=not args.fast, make_figures=not args.no_figures)
        # kNN probe alongside the parametric ones: it is the protocol the
        # historical Q1 numbers and MPC's Table 1 both used, so it is the only
        # metric directly comparable to what is already on record.
        try:
            import metrics as M
            rep["knn"] = {"accuracy": float(M.knn_accuracy(
                Ztr, np.asarray(ytr)[itr], Zte, np.asarray(yte)[ite]))}
        except Exception as e:
            rep["knn"] = {"error": f"{type(e).__name__}: {e}"}

        rep["tag"] = tag
        json.dump(rep, open(os.path.join(run_dir, "probes.json"), "w"), indent=2)
        rows.append(P.summarise(rep))

        s = rep["scene"]["regression"]
        print(f"  class probe   linear {rep['scene']['classification']['linear']['accuracy']:.4f}"
              + (f"  mlp {rep['scene']['classification']['mlp']['accuracy']:.4f}"
                 if "mlp" in rep["scene"]["classification"] else ""), flush=True)
        for k in ("flux", "size_rms", "ellipticity", "concentration"):
            row = s[k]
            extra = f"  mlp r={row['mlp']['r']:.3f}" if "mlp" in row else ""
            print(f"  {k:14s} linear r={row['linear']['r']:.3f} "
                  f"nmse={row['linear']['nmse']:.3f}{extra}", flush=True)
        print(f"  decode  ssim={rep['decode']['ssim']:.4f} "
              f"(floor {rep['decode_floor']['ssim']:.4f}, "
              f"gain {rep['decode']['ssim_gain_over_floor']:+.4f})  "
              f"psnr={rep['decode']['psnr']:.2f}", flush=True)
        if "accuracy" in rep.get("knn", {}):
            print(f"  knn probe     {rep['knn']['accuracy']:.4f}", flush=True)
        if "augmentation" in rep:
            print("  -- augmentation invariance (HIGH nmse = invariance held) --",
                  flush=True)
            a = rep["augmentation"]["regression"]
            for k in sorted(a):
                print(f"  {k:18s} linear r={a[k]['linear']['r']:+.3f} "
                      f"nmse={a[k]['linear']['nmse']:.3f}", flush=True)

    if rows:
        import csv
        out = "probes_summary.csv"
        keys = sorted({k for r in rows for k in r})
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag"] + [k for k in keys if k != "tag"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    main()
