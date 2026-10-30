import dataclasses as dc, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import data as D, diag_saccade as S
from config import DataCfg, data_preset

# Gates for running the predictors on MNIST and CIFAR-10 (plan_datasets_precision.md).  Run from Q1-PhaseA on the host
# after the datasets are in ../data.  All must pass before any training.
OK = [True]
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def loaders():
    for name, ntr, shape in (("mnist", 15675, (28, 28, 1)), ("cifar10", 15675, (32, 32, 3))):   # matched to Galaxy
        cfg = data_preset(name)
        Xtr, ytr, Xte, yte = D.load_dataset(cfg)
        ok = (Xtr.shape == (ntr,) + shape and Xte.shape == (2000,) + shape and Xtr.dtype == np.float32
              and ytr.shape == (ntr,) and yte.shape == (2000,))
        check(f"{name}: shapes and dtype", ok, f"{Xtr.shape} {Xte.shape} {Xtr.dtype}")
        check(f"{name}: pixel range [0, 1], not all one value", Xtr.min() >= 0 and Xtr.max() <= 1 and Xtr.std() > 0.1,
              f"min {Xtr.min():.2f} max {Xtr.max():.2f} std {Xtr.std():.3f}")
        cnt = np.bincount(ytr, minlength=10)
        check(f"{name}: 10 classes, each 5-15% of train; test covers all 10", len(cnt) == 10 and cnt.min() > 0.05 * ntr
              and cnt.max() < 0.15 * ntr and len(np.unique(yte)) == 10, str(cnt.tolist()))
        X2 = D.load_dataset(cfg)[0]
        check(f"{name}: loading is deterministic", np.array_equal(Xtr[:500], X2[:500]))
        full = D.load_dataset(dc.replace(cfg, n_train=-1, n_test=-1))
        check(f"{name}: subsets are drawn from the full sets ({len(full[0])} train, {len(full[2])} test)",
              len(full[0]) == {"mnist": 60000, "cifar10": 50000}[name] and len(full[2]) == 10000)
        tl = np.bincount(yte, minlength=10)
        check(f"{name}: random test subset (each class 6-14% of the 2000)", tl.min() > 120 and tl.max() < 280, str(tl.tolist()))
        if name == "cifar10":                    # class 0 (airplane) images are bluer than class 7 (horse) on average
            b = Xtr[ytr == 0][..., 2].mean() - Xtr[ytr == 7][..., 2].mean()
            check("cifar10: channel order RGB (airplanes bluer than horses)", b > 0.05, f"blue diff {b:.3f}")
            sm = float(np.mean(np.abs(Xtr[:500, :, 1:] - Xtr[:500, :, :-1])))
            check("cifar10: spatially smooth images (a scrambled layout gives ~0.25)", sm < 0.1, f"{sm:.3f}")
        # views: the encoder input dimension matches the preset
        Zin, bx, pr = D.make_views_labeled(jax.random.PRNGKey(0), jnp.asarray(Xtr[:16]), cfg)
        check(f"{name}: view dim == cfg.input_dim ({cfg.input_dim})", Zin.shape == (cfg.n_views, 16, cfg.input_dim), str(Zin.shape))
        if name == "mnist":
            check("mnist: no rotation and no flip in the views", float(np.max(np.abs(np.asarray(pr)[..., 0]))) == 0.0
                  and float(np.max(np.asarray(pr)[..., 1])) == 0.0)


def presets():
    check("galaxy10 preset == default DataCfg", data_preset("galaxy10") == DataCfg())
    p = os.path.join(HERE, "runs/bp_sigreg_etafix_s0/config.json")
    if os.path.exists(p):
        stored = json.load(open(p))["data"]; mine = dc.asdict(DataCfg())
        mine["crop_area"] = list(mine["crop_area"])
        check("galaxy10 preset == the Galaxy backbone's stored data config", stored == mine)
    else:
        check("Galaxy backbone config found", False, p)
    check("mnist preset grayscale, input_dim 1024", data_preset("mnist").grayscale and data_preset("mnist").input_dim == 1024)
    check("cifar10 preset colour, input_dim 3072, rotation 15 deg", data_preset("cifar10").input_dim == 3072
          and abs(data_preset("cifar10").max_rotation - np.pi / 12) < 1e-6)


def corruption():
    for H, C in ((28, 1), (32, 3), (69, 3)):
        X = np.ones((200, H, H, C), np.float32)
        Xo, occ = S.corrupt_images(X, "occlude")
        frac = float(np.mean(np.all(Xo == 0, -1)))
        check(f"{H}px: occluder covers 23-27% of the image", 0.23 <= frac <= 0.27, f"{frac:.3f}")
        Xn, _ = S.corrupt_images(0.5 * X, "noise")
        check(f"{H}px: noise clipped to [0, 1] with sd ~0.2", Xn.min() >= 0 and Xn.max() <= 1 and 0.17 < Xn.std() < 0.21, f"sd {Xn.std():.3f}")
        # occluded_frac(Hh=H) vs the black-pixel fraction of unrotated whole views cut from the occluded image
        ks = jax.random.split(jax.random.PRNGKey(3), 200)
        v, bx, _ = jax.vmap(lambda k, im: D._one_view(k, im, 32, 0.4, 1.0, 0.0, False, 0.0, 0.0))(ks, jnp.asarray(Xo))
        bx = np.asarray(bx); px = np.mean(np.all(np.asarray(v) < 1e-6, -1).reshape(200, -1), 1)
        pred = S.occluded_frac(bx[:, 1], bx[:, 0], bx[:, 2], occ, Hh=H)
        err = float(np.mean(np.abs(px - pred)))
        check(f"{H}px: occluded_frac(Hh={H}) matches the black-pixel fraction (mean abs err < 0.03)", err < 0.03, f"{err:.3f}")


def ds_dir():                                    # scripts run inside Q1-PhaseA/ds/<name>, data via the ds/data symlink
    with tempfile.TemporaryDirectory(dir=HERE) as t:
        os.makedirs(os.path.join(t, "x")); os.symlink(os.path.abspath(os.path.join(HERE, "..", "data")), os.path.join(t, "data"))
        cwd = os.getcwd(); os.chdir(os.path.join(t, "x"))
        try:
            Xtr = D.load_dataset(data_preset("mnist"))[0]
            check("dataset loads from inside a ds/<name> working directory", len(Xtr) == 15675)
        finally:
            os.chdir(cwd)


def galaxy_reproduction():                      # dataset-generic test_views/encode on Galaxy == stored latents (DS-7b)
    import pred_eval as PE
    from run_probes import load_run
    cfg, p = load_run(os.path.join(HERE, S.BACKBONE))
    check("Galaxy backbone data config == data_preset('galaxy10')", cfg.data == data_preset("galaxy10"))
    cwd = os.getcwd(); os.chdir(HERE)
    try:
        _, _, Xte, _ = D.load_dataset(cfg.data); L = np.load("pred_latents.npz")
        z = PE.encode(p, cfg.enc, PE.test_views(cfg.data, np.asarray(Xte[:256])))
        err = float(np.max(np.abs(z - L["bp_emb_te"][:256])) / np.max(np.abs(L["bp_emb_te"][:256])))
        check("Galaxy: re-encoded test views reproduce the stored latents (rel err < 1e-3)", err < 1e-3, f"{err:.1e}")
    finally:
        os.chdir(cwd)


def galaxy_regression():                         # the new whole-only path must reproduce findings/32 exactly on Galaxy
    out = os.path.join(tempfile.mkdtemp(), "reg.json")
    r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "diag_saccade2.py"), "--whole-only", "--depths", "10",
                        "--no-diag", "--require-all", "--seeds", "1", "--vmax", "6", "--out", out], cwd=HERE, capture_output=True, text=True)
    check("Galaxy whole-only run exits 0", r.returncode == 0, r.stderr[-400:] if r.returncode else "")
    if r.returncode:
        return
    new = {(x["model"], x["cond"], x["input"], x["pool"], x["K"]): x for x in json.load(open(out))["rows"]}
    old = {(x["model"], x["cond"], x["input"], x["pool"], x["K"]): x
           for x in json.load(open(os.path.join(HERE, "..", "findings/32_saccades_precision_followup/diag_saccade2.json")))["rows"]}
    common = [k for k in new if k in old]
    d_n = max(abs(new[k]["nmse"] - old[k]["nmse"]) / abs(old[k]["nmse"]) for k in common)
    d_k = max(abs(new[k]["knn"] - old[k]["knn"]) for k in common)
    models = {k[0] for k in common}
    check(f"Galaxy regression: {len(common)} rows ({sorted(models)}) reproduce findings/32", len(common) >= 60
          and models == {"G_d10_s1", "F_d10_s1", "BPg_d10_s1", "BPf_d10_s1"} and d_n < 1e-4 and d_k < 1e-3,
          f"max rel nmse diff {d_n:.1e}, max kNN diff {d_k:.1e}")


def main():
    presets(); loaders(); corruption(); ds_dir(); galaxy_reproduction(); galaxy_regression()
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
