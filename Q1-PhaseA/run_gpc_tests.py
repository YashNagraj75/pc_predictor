import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
import pred_gpc as G, pred_ff as PF, pred_eval as PE
from run_pred_tests import corrupted_latents, CORR

T_LIST = (5, 10, 25, 50, 100, 200, 400, 2000)
T_SETTLED = 2000


# Width-control and precision arms reuse the step scale of their parent arm
# (declared in the prereg).
def parent_key(a):
    return f"gpc_{a.rule}_d{a.depth}_{a.start}"


# Degenerate-route checks on test k=2: tau-only, shuffled s,
# per-level energy shares, convergence.
def diagnostics(p, am, cfg, zte, tte, rng_seed=0):
    n = len(zte)
    seen, tgt = PE.k_cases(n, 2)
    ii = np.arange(n)[:, None]
    Zs, Ts, Tt, Zt = (jnp.asarray(x) for x in (zte[ii, seen], tte[ii, seen], tte[ii, tgt], zte[ii, tgt]))
    out = {}
    nm = lambda P: float(jnp.mean(jnp.sum((P - Zt) ** 2, -1)) / jnp.mean(jnp.sum(Zt ** 2, -1)))
    for T, tag in ((cfg.T, "Ttrain"), (T_SETTLED, "settled")):
        f = eqx.filter_jit(lambda p, am, Zs, Ts: G.infer_s(p, am, Zs, Ts, cfg, T))
        s, s0, hist, rel = [], [], [], []
        for i in range(0, n, 500):
            r = f(p, am, Zs[i:i + 500], Ts[i:i + 500]); s.append(r[0]); s0.append(r[1]); rel.append(float(r[3]))
        s, s0 = jnp.concatenate(s), jnp.concatenate(s0)
        P = G.topdown(p, s, G.phi(Tt), cfg.H)[1]
        perm = np.random.default_rng(rng_seed).permutation(n)
        out[tag] = {"nmse": nm(P), "nmse_shuffled_s": nm(G.topdown(p, s[perm], G.phi(Tt), cfg.H)[1]),
                    "s_norm": float(jnp.mean(jnp.linalg.norm(s, axis=-1))), "rel_last_step_s": float(np.mean(rel)),
                    "amort_rel_dist": float(jnp.mean(jnp.linalg.norm(s - s0, axis=-1) / (jnp.linalg.norm(s, axis=-1) + 1e-8))) if cfg.start == "amort" else None}
        if cfg.rule == "pc":   # per-level energy shares at the settled seen-view state
            sp, xs, _, _ = G.pc_infer(p, G.start_s(p, am, Zs[:500], Ts[:500], cfg), Zs[:500], G.phi(Ts[:500]), cfg, T)
            e = [float(v) for v in G.level_energies(p, sp, xs, Zs[:500], G.phi(Ts[:500]), cfg.H)]
            out[tag]["level_energy_share"] = [v / sum(e) for v in e]
    out["nmse_tau_only"] = nm(G.topdown(p, jnp.zeros((n, G.M)), G.phi(Tt), cfg.H)[1])
    out["nmse_copy_mean_seen"] = nm(jnp.repeat(jnp.mean(Zs, 1, keepdims=True), Zt.shape[1], 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="pc"); ap.add_argument("--depth", type=int, default=3); ap.add_argument("--start", default="zero")
    ap.add_argument("--backbone", default="bp"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=0); ap.add_argument("--hid-prec", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=None)   # alm only: force the penalty (ablation); default = selected
    ap.add_argument("--alpha", type=float, default=1.0)  # alm only: 0 = rho-matched plain-PC control (same energy, no duals)
    ap.add_argument("--ckpt", action="store_true")       # also save weights at steps 1000 and 2000
    ap.add_argument("--T2", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--gsel", default="pred_runs/gpc_selected.json")
    ap.add_argument("--outdir", default="pred_runs/tests")
    a = ap.parse_args()
    ztr, ttr, zte, tte, *_ = PF.load("pred_latents.npz", a.backbone, "emb")
    raw = np.load("pred_latents.npz")[f"{a.backbone}_emb_tr"].reshape(-1, 128)
    zstats = (raw.mean(0), raw.std(0) + 1e-6)
    sel = json.load(open(a.gsel))
    c, hp, alpha = sel[parent_key(a)]["c"], a.hid_prec, 0.0
    if a.rule == "alm":
        alpha = a.alpha
        hp = sel[parent_key(a)]["rho"] if a.rho is None else a.rho
        c = sel[parent_key(a)]["c"] if a.rho is None else sel[parent_key(a)]["best_c_by_rho"][str(float(a.rho))]
    cfg = G.GCfg(backbone=a.backbone, rule=a.rule, depth=a.depth, start=a.start, width=a.width, hid_prec=hp,
                 seed=a.seed, c=c, steps=a.steps, alpha=alpha, **({"T2": a.T2} if a.T2 else {}))
    os.makedirs(a.outdir, exist_ok=True)
    out = f"{a.outdir}/{cfg.name}_{a.backbone}_s{a.seed}.json"
    if os.path.exists(out):
        print("EXISTS -- skipping", out, flush=True); sys.exit(0)
    t0 = time.time()
    import dataclasses
    ck = (out[:-5], {1000, 2000}) if a.ckpt else None
    attempts = []
    for c_try in (c, c / 2, c / 4):          # retry on divergence OR blow-up (review N6); c_used is recorded and flagged
        cfg = dataclasses.replace(cfg, c=c_try)
        p, am, log, div = G.train(cfg, ztr, ttr, verbose=True, ckpt=ck)
        ex = (not div) and G.exploded(log)
        attempts.append({"c": c_try, "diverged": div, "exploded": ex, "steps_done": log[-1]["step"] if log else 0})
        if not (div or ex):
            break
    c_used = cfg.c
    if div or ex:
        G.dump_json({"arm": cfg.name, "backbone": a.backbone, "seed": a.seed, "diverged": True, "attempts": attempts,
                     "exploded": G.exploded(log), "train_log": log}, out.replace(".json", ".diverged.json"))
        print("DIVERGED -- no result written", flush=True); sys.exit(3)
    eqx.tree_serialise_leaves(out[:-5] + ".eqx", p)
    rows = []
    predict = G.make_predict(p, am, cfg)
    settled = G.make_predict(p, am, cfg, T=T_SETTLED)
    rows.append({"test": "pairs", "condition": "clean", "k": 1, "T": cfg.T, **PE.eval_pairs(predict, zte, tte)})
    cases = {k: PE.k_cases(len(zte), k) for k in (1, 2, 3)}
    for k in (1, 2, 3):
        rows.append({"test": "k_seen", "condition": "clean", "k": k, "T": cfg.T, **PE.run_cases(predict, zte, tte, *cases[k])})
        rows.append({"test": "k_seen_settled", "condition": "clean", "k": k, "T": T_SETTLED, **PE.run_cases(settled, zte, tte, *cases[k])})
    Ts = ([0] if a.start == "amort" else []) + list(T_LIST)
    for T in Ts:
        rows.append({"test": "T_inf", "condition": "clean", "k": 1, "T": T, **PE.run_cases(G.make_predict(p, am, cfg, T=T), zte, tte, *cases[1])})
    if a.backbone == "bp":
        for k in (1, 2):
            seen, tgt = cases[k]
            for cs in (0, 1, 2):                                    # three corruption draws (review D9)
                corr = corrupted_latents(a.backbone, zstats, seen[:, 0], seed=cs)
                for (kind, lvl), zc in corr.items():
                    zs = zte.copy(); zs[np.arange(len(zte)), seen[:, 0]] = zc[np.arange(len(zte)), seen[:, 0]]
                    rows.append({"test": "robust", "condition": f"{kind}={lvl}", "corr_seed": cs, "k": k, "T": cfg.T,
                                 **PE.run_cases(predict, zte, tte, seen, tgt, zseen=zs)})
    diag = diagnostics(p, am, cfg, zte, tte)
    bad = [f"{r['test']}/{r['condition']}/k{r['k']}/T{r['T']}" for r in rows if not np.isfinite(r.get("nmse", np.nan))]
    n_gen = G.n_params(p); n_am = G.n_params(eqx.filter(am, eqx.is_inexact_array)) if am is not None else 0
    G.dump_json({"method": cfg.name, "backbone": a.backbone, "seed": a.seed, "c_selected": c, "c_used": c_used, "c_changed": c_used != c, "attempts": attempts, "nonfinite_rows": bad,
                 "rows": rows, "diagnostics": diag, "n_params_generator": n_gen, "n_params_amortiser": n_am,
                 "cfg": cfg.__dict__, "seconds": time.time() - t0, "train_log": log}, out)
    for r in rows:
        if r["test"] in ("k_seen", "k_seen_settled") or (r["test"] == "robust" and r["corr_seed"] == 0):
            print(f"{cfg.name} {a.backbone} s{a.seed} {r['test']:15s} {r['condition']:12s} k={r['k']} T={r['T']} nmse {r['nmse']:.4f} retr1 {r['retr1']:.4f}", flush=True)
    print("DIAG", json.dumps(G._clean(diag)), flush=True)


if __name__ == "__main__":
    main()
