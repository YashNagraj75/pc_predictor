import argparse, copy, dataclasses, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np, equinox as eqx
import data as D, sigreg as S, sparse as SP, pcalm as PA, train as TR
from config import PCCfg, RunCfg, SparseCfg

W = lambda g, l: g[0][l].layers[1].weight


def cfg_of(sel, K, ws=None):
    st = sel["sets"][str(K)]
    sp = SparseCfg(case="D", heads=tuple(st["heads"]), gammas=tuple(st["gammas"]),
                   head_ws=tuple(ws if ws is not None else st["head_ws"]), beta_g=sel["beta_g"],
                   T_test=sel["T_test"], beta_r=sel.get("beta_r", 0.0), code_k=sel.get("code_k", 1))
    return RunCfg(arm="pc", use_sigreg=True, pc=PCCfg(T=16, eta_h=0.0625, alpha=1.0), sparse=sp)


def ratios(params, cfg, Zin, A, kg):
    out = {}
    zero = dataclasses.replace(cfg, sparse=dataclasses.replace(cfg.sparse, head_ws=(0.0,) * len(cfg.sparse.heads)))
    for arm in ("pc", "bp"):
        c, c0 = dataclasses.replace(cfg, arm=arm), dataclasses.replace(zero, arm=arm)
        if arm == "pc":
            gf, _ = PA.grads(params, Zin, A, c, kg); gt, _ = PA.grads(params, Zin, A, c0, kg)
        else:
            gf, _ = TR.bp_grads(params, Zin, A, c); gt, _ = TR.bp_grads(params, Zin, A, c0)
        top = max(cfg.sparse.heads)
        h = [float(jnp.linalg.norm(W(gf, l) - W(gt, l))) for l in range(len(params[0]))]
        t = [float(jnp.linalg.norm(W(gt, l))) for l in range(len(params[0]))]
        agg = float(np.sqrt(sum(x ** 2 for x in h[:top + 1])) / np.sqrt(sum(x ** 2 for x in t[:top + 1])))
        out[arm] = {"per_layer": [a / b for a, b in zip(h, t)], "upto_top": agg}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sel"); ap.add_argument("--K", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--write-pc", default=None); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    sel = json.load(open(a.sel))
    Xtr = D.load_dataset(RunCfg().data)[0]
    idx = np.random.default_rng(0).choice(len(Xtr), a.n, replace=False)
    kv, ka = jax.random.split(jax.random.PRNGKey(123))
    Zin, _ = D.make_views(kv, jnp.asarray(Xtr[idx[:a.batch]]), RunCfg().data)
    A = S.sample_directions(ka, RunCfg().enc.embed_dim, RunCfg().sig.num_slices)
    kg = jax.random.PRNGKey(7)
    res, newsel = {}, copy.deepcopy(sel)
    for K in a.K:
        cfg = cfg_of(sel, K)
        p = SP.init_mu(SP.init_params(jax.random.PRNGKey(0), cfg, cfg.data.input_dim), SP.mu_rows(Xtr, cfg), cfg)
        r = ratios(p, cfg, Zin, A, kg)
        f = lambda v: " ".join(f"{x:.3f}" for x in v)
        print(f"K={K}  BP head/task upto top {r['bp']['upto_top']:.3f} | per layer {f(r['bp']['per_layer'])}", flush=True)
        print(f"K={K}  PC head/task upto top {r['pc']['upto_top']:.3f} | per layer {f(r['pc']['per_layer'])}", flush=True)
        res[str(K)] = {"default": r}
        if a.write_pc:
            target = sel["grad_ratio"]
            s = target / r["pc"]["upto_top"]
            ws = [w * s for w in sel["sets"][str(K)]["head_ws"]]
            r2 = ratios(p, cfg_of(sel, K, ws), Zin, A, kg)       # re-measure (relaxation is nonlinear)
            s2 = s * target / r2["pc"]["upto_top"]
            ws = [w * s2 / s for w in ws]
            r3 = ratios(p, cfg_of(sel, K, ws), Zin, A, kg)
            print(f"K={K}  PC rescaled x{s2:.4f}: PC head/task upto top {r3['pc']['upto_top']:.3f} | per layer {f(r3['pc']['per_layer'])}", flush=True)
            newsel["sets"][str(K)]["head_ws"] = ws
            newsel["sets"][str(K)]["pc_dose_scale"] = s2
            res[str(K)]["pc_matched"] = {"scale": s2, "ratios": r3}
    if a.write_pc:
        newsel["rule"] += f"; PC arm head_ws rescaled so PC's own head/task gradient ratio is {sel['grad_ratio']} (diag_dose.py)"
        json.dump(newsel, open(a.write_pc, "w"), indent=1)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
