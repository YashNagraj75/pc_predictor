import dataclasses as dc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, encoder as E, pred_eval as PE, pred_gpc as G, pred_ffres as RF
from pred_data import tau_features
from run_probes import load_run

# S0 of the saccade plan: how small can a glimpse crop be before the frozen backbone's latents stop being usable?
# Inference only.  Per crop area: OOD distance of the latent, kNN class accuracy, and how well the trained PC
# predictors (G: amortised generative PC-ALM d3/d10; F: residual ff PC-ALM d10 rho16) predict the 4 stored target
# views from that single crop.  No BP model is touched (user rule 2026-10-27).
OUT = sys.argv[1] if len(sys.argv) > 1 else "pred_runs/diag_scale.json"
AREAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
RUNS = ["runs/bp_sigreg_etafix_s0", "runs/bp_sigreg_e400"]


def pick_backbone(Xte, raw_te):
    for r in RUNS:
        if not os.path.exists(r):
            continue
        cfg, p = load_run(r)
        pix = PE.test_views(cfg.data, Xte[:512])
        z = PE.encode(p, cfg.enc, pix)
        err = float(np.max(np.abs(z - raw_te[:512])) / np.max(np.abs(raw_te[:512])))
        print("backbone", r, "re-encode rel err", err, flush=True)
        if err < 1e-3:
            return r, cfg, p, err
    raise SystemExit("no backbone reproduces the stored latents")


def crops(cfg, X, area, key, chunk=500):
    one = jax.vmap(lambda k, im: D._one_view(k, im, cfg.data.img_size, area, area, 0.0, False, 0.0, 0.0))
    pix, tau = [], []
    for i in range(0, len(X), chunk):
        ks = jax.random.split(jax.random.fold_in(key, i), len(X[i:i + chunk]))
        v, bx, pr = one(ks, jnp.asarray(X[i:i + chunk]))
        if cfg.data.grayscale and v.shape[-1] == 3:
            v = v.mean(-1, keepdims=True)
        pix.append(np.asarray(v).reshape(v.shape[0], -1)); tau.append(tau_features(bx, pr))
    return np.concatenate(pix), np.concatenate(tau)


def load_G(depth, seed):
    base = f"pred_runs/tests_sep_fix/gpc_sep_d{depth}_amort_hp16_a1_k1_amort_c1_T2200_bp_s{seed}"
    d = json.load(open(base + ".json"))
    names = {f.name for f in dc.fields(G.GCfg)}
    cfg = G.GCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["cfg"].items() if k in names})
    p = eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), cfg))
    am = eqx.tree_deserialise_leaves(base + "_amort.eqx", G.init_amort(jax.random.PRNGKey(0)))
    return G.make_predict(p, am, cfg), d["rows"]


def load_F(seed):
    rc = RF.RCfg(arm="alm", depth=10, seed=seed, T=400, alpha=1.0, lr=0.01, rho=16.0)
    path = f"pred_runs/ffres_fix/ffres_almR16_d10_s{seed}.eqx"
    return RF.make_pair_predict(RF.load_model(path, rc), rc)


def nmse_all_targets(predict, zc, tc, zte, tte, batch=500):
    n = len(zte); P = []
    for i in range(0, n, batch):
        P.append(np.asarray(predict(zc[i:i + batch, None], tc[i:i + batch, None], tte[i:i + batch])))
    P = np.concatenate(P)
    img = np.repeat(np.arange(n), 4); tv = np.tile(np.arange(4), n)
    return PE._metrics(P.reshape(-1, P.shape[-1]), zte.reshape(-1, zte.shape[-1]), img, tv, zte)


def knn(ztr, ytr, z, y, k=20):
    a = ztr / np.linalg.norm(ztr, axis=1, keepdims=True); b = z / np.linalg.norm(z, axis=1, keepdims=True)
    pred = []
    for i in range(0, len(b), 500):
        idx = np.argpartition(-(b[i:i + 500] @ a.T), k, axis=1)[:, :k]
        pred.append([np.bincount(r, minlength=10).argmax() for r in ytr[idx]])
    return float(np.mean(np.concatenate(pred) == y))


def main():
    L = np.load("pred_latents.npz")
    raw_tr, raw_te = L["bp_emb_tr"], L["bp_emb_te"]
    ytr, yte = L["ytr"], L["yte"]
    mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (raw_te - mu) / sd, L["tau_te"].astype(np.float32)
    cfg0, _ = load_run(RUNS[0]) if os.path.exists(RUNS[0]) else load_run(RUNS[1])
    _, _, Xte, _ = D.load_dataset(cfg0.data)
    Xte = np.asarray(Xte)
    assert len(Xte) == len(zte), (len(Xte), len(zte))
    run, cfg, params, rerr = pick_backbone(Xte, raw_te)
    ztr_n = ((raw_tr - mu) / sd).reshape(-1, 128)
    C = np.cov(ztr_n.T) + 1e-3 * np.eye(128); Ci = np.linalg.inv(C); m0 = ztr_n.mean(0)
    maha = lambda z: np.sqrt(np.einsum("nd,de,ne->n", z - m0, Ci, z - m0))
    ref_d = maha(zte.reshape(-1, 128)); p99 = float(np.percentile(ref_d, 99))
    res = {"backbone": run, "reencode_rel_err": rerr, "areas": AREAS, "ref": {
        "maha_median": float(np.median(ref_d)), "maha_p99": p99,
        "knn_view0": knn(ztr_n.reshape(raw_tr.shape)[:, 0], ytr, zte[:, 0], yte)}, "rows": [],
        "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "XLA_FLAGS")}}
    preds = {f"G_d{d}_s{s}": load_G(d, s) for d in (3, 10) for s in (0, 1, 2)}
    res["ref"]["G_k1_stored"] = {k: next(r["nmse"] for r in v[1] if r["test"] == "k_seen" and r.get("k") == 1) for k, v in preds.items()}
    preds = {k: v[0] for k, v in preds.items()}
    preds.update({f"F_d10_s{s}": load_F(s) for s in (0, 1, 2)})
    for a in AREAS:
        t0 = time.time()
        pix, tc = crops(cfg, Xte, a, jax.random.PRNGKey(int(a * 1000) + 7))
        zc = (PE.encode(params, cfg.enc, pix[:, None])[:, 0] - mu) / sd
        d = maha(zc)
        row = {"area": a, "scale": float(np.sqrt(a)), "maha_median": float(np.median(d)),
               "frac_above_ref_p99": float(np.mean(d > p99)), "knn": knn(ztr_n.reshape(raw_tr.shape)[:, 0], ytr, zc, yte),
               "z_norm_ratio": float(np.median(np.linalg.norm(zc, axis=1)) / np.median(np.linalg.norm(zte[:, 0], axis=1)))}
        for k, f in preds.items():
            row[k] = nmse_all_targets(f, zc, tc, zte, tte)
        row["sec"] = round(time.time() - t0, 1)
        res["rows"].append(row)
        print("ROW", json.dumps({k: (round(v, 4) if isinstance(v, float) else (round(v["nmse"], 4) if isinstance(v, dict) else v)) for k, v in row.items()}), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
