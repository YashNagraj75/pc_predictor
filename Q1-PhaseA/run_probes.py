"""Run the latent-quality suite on one or more SAVED runs.

    ./.venv/bin/python run_probes.py runs/bp_sigreg runs/pc_sigreg_T16_coupled
    ./.venv/bin/python run_probes.py runs_smoke/* --fast

Reads each run directory's `config.json` + `params.eqx`, rebuilds the encoder,
embeds a fixed evaluation split, and writes `probes.json` + two figures INTO
that run directory.  Nothing is re-trained.

Why a separate entry point rather than folding this into train.py's
`evaluate`: the decoder and the MLP probes cost real time, so they should not
run at every eval step -- and a post-hoc runner means the suite applies to runs
that finished before the suite existed, as long as their params were saved.
"""
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
from config import DataCfg, EncoderCfg, OptCfg, PCCfg, RunCfg, SIGRegCfg


def cfg_from_dict(d):
    """Rebuild RunCfg from a saved config.json (nested dataclasses, and
    to_json wrote tuples as lists)."""
    def mk(kls, sub):
        fields = {f.name: f.type for f in kls.__dataclass_fields__.values()}
        kw = {}
        for k, v in sub.items():
            if k not in fields:
                continue
            kw[k] = tuple(v) if isinstance(v, list) else v
        return kls(**kw)
    top = {k: v for k, v in d.items()
           if k not in ("data", "enc", "sig", "pc", "opt")}
    return RunCfg(**top, data=mk(DataCfg, d["data"]), enc=mk(EncoderCfg, d["enc"]),
                  sig=mk(SIGRegCfg, d["sig"]), pc=mk(PCCfg, d["pc"]),
                  opt=mk(OptCfg, d["opt"]))


def load_run(run_dir):
    cfg = cfg_from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    skeleton = E.init(jax.random.PRNGKey(cfg.opt.seed), cfg.enc, cfg.data.input_dim)
    params = eqx.tree_deserialise_leaves(os.path.join(run_dir, "params.eqx"),
                                         skeleton)
    return cfg, params


def embed_split(params, cfg, X, n, seed=0):
    """Deterministic centre view -> (Z, imgs_as_seen).

    `imgs_as_seen` is the RESHAPED encoder input, not the raw dataset image:
    probe targets have to be measured on the pixels the encoder actually got,
    including the resize to cfg.data.img_size and any grayscaling.
    """
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    xb = D.eval_batch(jax.numpy.asarray(X[idx]), cfg.data)
    z, _ = E.forward(params, xb, cfg.enc)
    c = 1 if cfg.data.grayscale else 3
    imgs = np.asarray(xb).reshape(-1, cfg.data.img_size, cfg.data.img_size, c)
    return np.asarray(z), imgs, idx


def main():
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
        if not os.path.exists(os.path.join(run_dir, "params.eqx")):
            print(f"[skip] {run_dir}: no params.eqx", flush=True)
            continue
        cfg, params = load_run(run_dir)
        Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
        Ztr, Itr, itr = embed_split(params, cfg, Xtr, args.n_train, seed=0)
        Zte, Ite, ite = embed_split(params, cfg, Xte, args.n_test, seed=1)

        tag = os.path.basename(os.path.normpath(run_dir))
        print(f"\n=== {tag}  Z{Ztr.shape} -> img{Itr.shape[1:]} ===", flush=True)
        rep = P.run_suite(
            Ztr, Zte, Itr, Ite, np.asarray(ytr)[itr], np.asarray(yte)[ite],
            out_dir=run_dir, tag="",
            decoder_steps=400 if args.fast else args.decoder_steps,
            probe_mlp=not args.fast, make_figures=not args.no_figures)
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
