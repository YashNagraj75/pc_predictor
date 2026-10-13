"""Apply the pre-registered tests (pred_sdpc_prereg.json) to one trained predictor.

    ./.venv/bin/python run_pred_tests.py sdpc  --backbone bp --seed 0 [--sel pred_runs/sdpc_selected.json]
    ./.venv/bin/python run_pred_tests.py bpset --backbone bp --seed 0
    ./.venv/bin/python run_pred_tests.py ff    --backbone bp --seed 0 --arm bp|pcA      (k = 1 tests only)
    ./.venv/bin/python run_pred_tests.py copy  --backbone bp
Writes pred_runs/tests/<method>_<backbone>_s<seed>.json : rows of {test, condition, k, T, nmse, cos, retr1}.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
import pred_ff as PF, pred_eval as PE, pred_sdpc as S, pred_bpset as B

CORR = [("noise", 0.1), ("noise", 0.2), ("occlude", 0.25)]


def corrupted_latents(backbone, zstats, seen_first):
    """Re-encode the test views with the first seen view corrupted; standardise like pred_ff.load."""
    import data as D, encoder as E
    from run_probes import load_run
    d0 = np.load("pred_latents.npz")
    run = str(d0["backbone_run"]) if "backbone_run" in d0 else "runs/bp_sigreg_e400"
    cfg, p_bp = load_run(run)          # the SAME backbone the stored latents came from
    p = p_bp if backbone == "bp" else E.init(jax.random.PRNGKey(0), cfg.enc, cfg.data.input_dim)
    Xte = D.load_dataset(cfg.data)[2]
    pix = PE.test_views(cfg.data, Xte)
    d = np.load("pred_latents.npz")
    PE.check_reproduction(PE.encode(p, cfg.enc, pix), d[f"{backbone}_emb_te"])
    mu, sd = zstats
    out = {}
    for kind, lvl in CORR:
        z = PE.encode(p, cfg.enc, PE.corrupt(pix, seen_first, kind, lvl, seed=0))
        out[(kind, lvl)] = (z - mu) / sd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["sdpc", "bpset", "ff", "copy"])
    ap.add_argument("--backbone", default="bp"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="bp"); ap.add_argument("--sel", default="pred_runs/sdpc_selected.json")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    ztr, ttr, zte, tte, *_ = PF.load("pred_latents.npz", a.backbone, "emb")
    raw = np.load("pred_latents.npz")[f"{a.backbone}_emb_tr"].reshape(-1, 128)
    zstats = (raw.mean(0), raw.std(0) + 1e-6)
    t0 = time.time(); extra = {}
    if a.method == "sdpc":
        sel = json.load(open(a.sel))
        cfg = S.SCfg(backbone=a.backbone, seed=a.seed, lam=sel["lam"], beta=sel["beta"], lam_decay=sel["lam_decay"],
                     T_inf=sel["T_inf"], steps=a.steps or sel["steps"])
        Ds, log = S.train(cfg, ztr, ttr, verbose=True)
        predict = S.make_predict(Ds, cfg)
        extra = {"log": log, "n_params": S.n_params(Ds), "cfg": cfg.__dict__}
        Tsweep = {T: S.make_predict(Ds, cfg, T=T) for T in (5, 10, 25, 50, 100, 200)}
    elif a.method == "bpset":
        model = B.train(B.BCfg(seed=a.seed, width=384, steps=a.steps or 6000), ztr, ttr)
        predict = B.make_predict(model); extra = {"n_params": B.n_params(model)}; Tsweep = {}
    elif a.method == "ff":
        f = f"pred_runs/ff_{a.arm}_{a.backbone}_emb_s{a.seed}.eqx"
        predict = PF.make_pair_predict(PF.load_model(f, seed=a.seed)); Tsweep = {}
    else:
        predict = lambda Zs, Ts, Tt: np.repeat(np.asarray(Zs).mean(1, keepdims=True), np.asarray(Tt).shape[1], 1)
        Tsweep = {}
    ks = [1] if a.method == "ff" else [1, 2, 3]
    rows = [{"test": "pairs", "condition": "clean", "k": 1, **PE.eval_pairs(predict, zte, tte)}]
    cases = {k: PE.k_cases(len(zte), k) for k in ks}
    for k in ks:
        rows.append({"test": "k_seen", "condition": "clean", "k": k, **PE.run_cases(predict, zte, tte, *cases[k])})
    for T, pr in Tsweep.items():
        rows.append({"test": "T_inf", "condition": "clean", "k": 1, "T": T, **PE.run_cases(pr, zte, tte, *cases[1])})
    if a.backbone == "bp":
        for k in [k for k in ks if k <= 2]:
            seen, tgt = cases[k]
            corr = corrupted_latents(a.backbone, zstats, seen[:, 0])
            for (kind, lvl), zc in corr.items():
                zs = zte.copy(); zs[np.arange(len(zte)), seen[:, 0]] = zc[np.arange(len(zte)), seen[:, 0]]
                rows.append({"test": "robust", "condition": f"{kind}={lvl}", "k": k, **PE.run_cases(predict, zte, tte, seen, tgt, zseen=zs)})
    name = a.method if a.method != "ff" else f"ff{a.arm}"
    os.makedirs("pred_runs/tests", exist_ok=True)
    out = f"pred_runs/tests/{name}_{a.backbone}_s{a.seed}.json"
    json.dump({"method": name, "backbone": a.backbone, "seed": a.seed, "rows": rows, "seconds": time.time() - t0,
               **{k: v for k, v in extra.items() if k != "log"}, "train_log": extra.get("log")}, open(out, "w"), indent=1, default=str)
    for r in rows:
        print(f"{name:6s} {a.backbone} s{a.seed} {r['test']:7s} {r['condition']:12s} k={r['k']} T={r.get('T', '')!s:4s} "
              f"nmse {r['nmse']:.4f} cos {r['cos']:.4f} retr1 {r['retr1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
