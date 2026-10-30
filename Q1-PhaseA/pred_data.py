import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
import data as D, encoder as E
jax.config.update("jax_default_matmul_precision", "highest")   # same precision as the tests' re-encoding (review N6)
from run_probes import load_run

LAYERS = {"h3": 3, "h7": 7, "emb": None}


def tau_features(boxes, params):
    b, p = np.asarray(boxes, np.float32), np.asarray(params, np.float32)
    return np.concatenate([b, np.sin(p[..., :1]), np.cos(p[..., :1]), p[..., 1:4]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bp_sigreg_e400")
    ap.add_argument("--out", default="pred_latents.npz")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=-1)
    a = ap.parse_args()
    cfg, p_bp = load_run(a.run)
    p_rand = E.init(jax.random.PRNGKey(0), cfg.enc, cfg.data.input_dim)
    Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
    if a.n_train > 0:
        Xtr, ytr = Xtr[:a.n_train], ytr[:a.n_train]
    import equinox as eqx
    fwd = eqx.filter_jit(lambda p, x: E.forward(p, x, cfg.enc))
    out = {"ytr": np.asarray(ytr), "yte": np.asarray(yte), "backbone_run": np.array(a.run)}
    for split, X, key in (("tr", Xtr, 101), ("te", Xte, 202)):
        taus, lats = [], {f"{b}_{l}": [] for b in ("bp", "rand") for l in LAYERS}
        for i in range(0, len(X), a.chunk):
            k = jax.random.fold_in(jax.random.PRNGKey(key), i)
            Zin, boxes, params = D.make_views_labeled(k, jnp.asarray(X[i:i + a.chunk]), cfg.data)
            taus.append(tau_features(boxes, params))                 # (V, b, 9)
            for bname, p in (("bp", p_bp), ("rand", p_rand)):
                per_v = [fwd(p, Zin[v]) for v in range(Zin.shape[0])]
                for l, idx in LAYERS.items():
                    lats[f"{bname}_{l}"].append(np.stack([np.asarray(z if idx is None else hs[idx]) for z, hs in per_v]))
        out[f"tau_{split}"] = np.concatenate(taus, 1).transpose(1, 0, 2)          # (N, V, 9)
        for k_, v in lats.items():
            out[f"{k_}_{split}"] = np.concatenate(v, 1).transpose(1, 0, 2)        # (N, V, d)
        print(split, out[f"tau_{split}"].shape, out[f"bp_emb_{split}"].shape, flush=True)
    tmp = a.out[:-4] + ".tmp.npz"                                 # atomic: a killed job leaves no truncated file
    np.savez_compressed(tmp, **out); os.replace(tmp, a.out)
    print("wrote", a.out, os.path.getsize(a.out) // 2**20, "MB")


if __name__ == "__main__":
    main()
