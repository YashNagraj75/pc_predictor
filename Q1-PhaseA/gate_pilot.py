"""Pilot gate for one sparse PAIR (PC arm + its BP control), rule D5.

    ./.venv/bin/python gate_pilot.py runs/<pc_pilot_dir> runs/<bp_pilot_dir>
    exit 0 = pass, 1 = fail

Criteria (spec: 'Method - Sparse PC-ALM arms spec'), applied to BOTH arms:
  * every logged value finite, no DIVERGED marker
  * encoder gradient clipping fired on < 5 % of steps (for case C the encoder
    group; the dictionary group is unclipped by design)
  * DEPLOYED active fraction (eval rows, SP.health) at the last eval in [0.1, 0.3]
    -- the deployed network is what is probed, so that is what is gated
PC arm only:
  * act_growth = max_l ||z_l^(T)|| / max_k ||z_k^(0)|| <= 10 over the last 200 steps
Pair:
  * the two config.json files differ ONLY in `arm`, and the PC arm has alpha > 0
A failing pair is skipped and reported -- never re-tuned against the pilot.

Review history: the first version gated the PC arm alone and on the RELAXED
sparse variable; three reviewers flagged that the past clipping confound could
recur on the BP control and that relaxed != deployed sparsity.
"""
import json
import math
import os
import sys

LAST = 200
BAND = (0.1, 0.3)
# case D (wiki 'Method - Sparse side-head arm plan', D4): band per head, plus two
# collapse guards that cases B/C lacked
BAND_D = (0.05, 0.4)
SUPPORTS_MIN_FRAC = 0.5          # distinct active patterns >= 50 % of eval rows, every head
SPREAD_MIN_RATIO = 0.5           # embedding input spread >= 0.5 x its value at step 0


def _load(run_dir):
    log = json.load(open(os.path.join(run_dir, "log.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    return log, cfg


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def arm_report(run_dir, is_pc):
    log, cfg = _load(run_dir)
    clip = cfg["opt"]["grad_clip"]
    case = cfg.get("sparse", {}).get("case")
    is_C = case in ("C", "D")
    key = "grad_norm_enc" if is_C else "grad_norm_preclip"
    g = [r[key] for r in log if key in r]
    lf = [r["loss_ff"] for r in log if "loss_ff" in r]
    dep = [(r["step"], r["eval"]["sparse"]["active_frac"]) for r in log
           if isinstance(r.get("eval"), dict) and "sparse" in r["eval"]]
    info = {"run": run_dir, "n_steps": len(log),
            "clip_frac_enc": (sum(v > clip for v in g) / len(g)) if g else None,
            "deployed_active_frac": [round(v, 4) for _, v in dep],
            "deployed_active_steps": [s for s, _ in dep]}
    reasons = []
    if os.path.exists(os.path.join(run_dir, "DIVERGED")) or any(r.get("DIVERGED") for r in log):
        reasons.append("run diverged")
    vals = g + lf
    if is_pc:
        ag = [r["act_growth"] for r in log[-LAST:] if "act_growth" in r]
        rf = [r["active_frac"] for r in log[-LAST:] if "active_frac" in r]
        info["act_growth_max_last200"] = max(ag) if ag else None
        info["relaxed_active_frac_last200"] = (sum(rf) / len(rf)) if rf else None
        vals += ag
        if not ag:
            reasons.append("PC log has no act_growth")
        elif max(ag) > 10:
            reasons.append(f"activity growth {max(ag):.2f} > 10")
    if not g:
        reasons.append(f"log has no {key}")
    elif info["clip_frac_enc"] >= 0.05:
        reasons.append(f"encoder clipping fired on {100 * info['clip_frac_enc']:.1f}% of steps (>= 5%)")
    if not all(math.isfinite(v) for v in vals):
        reasons.append("non-finite values in the log")
    if case == "D":
        reasons += _gate_D(log, info)
    elif not dep:
        reasons.append("no deployed active fraction in eval rows")
    elif not BAND[0] <= dep[-1][1] <= BAND[1]:
        reasons.append(f"deployed active fraction {dep[-1][1]:.3f} outside {list(BAND)}")
    return reasons, info, cfg


def _gate_D(log, info):
    ev = [r["eval"]["sparse"] for r in log if isinstance(r.get("eval"), dict) and "sparse" in r["eval"]]
    if len(ev) < 2:
        return ["fewer than two eval rows with case-D health"]
    first, last = ev[0], ev[-1]
    heads = sorted(int(k[1:].split("_")[0]) for k in last if k.endswith("_active"))
    out = []
    info["head_active_last"] = {l: round(last[f"h{l}_active"], 4) for l in heads}
    info["head_supports_last"] = {l: last[f"h{l}_supports"] for l in heads}
    info["embed_spread_first_last"] = (first["embed_spread"], last["embed_spread"])
    for l in heads:
        a = last[f"h{l}_active"]
        if not BAND_D[0] <= a <= BAND_D[1]:
            out.append(f"head {l} deployed active fraction {a:.3f} outside {list(BAND_D)}")
        if last[f"h{l}_supports"] < SUPPORTS_MIN_FRAC * last["n_rows"]:
            out.append(f"head {l}: {last[f'h{l}_supports']} distinct supports < "
                       f"{SUPPORTS_MIN_FRAC:.0%} of {last['n_rows']} rows (collapse)")
    if last["embed_spread"] < SPREAD_MIN_RATIO * first["embed_spread"]:
        out.append(f"embedding spread {last['embed_spread']:.4g} < {SPREAD_MIN_RATIO} x "
                   f"initial {first['embed_spread']:.4g} (collapse)")
    return out


def gate(pc_dir, bp_dir):
    r_pc, i_pc, c_pc = arm_report(pc_dir, is_pc=True)
    r_bp, i_bp, c_bp = arm_report(bp_dir, is_pc=False)
    reasons = [f"PC: {x}" for x in r_pc] + [f"BP: {x}" for x in r_bp]
    if c_pc.get("arm") != "pc" or c_bp.get("arm") != "bp":
        reasons.append(f"arms are {c_pc.get('arm')}/{c_bp.get('arm')}, expected pc/bp")
    if not c_pc.get("pc", {}).get("alpha", 0) > 0:
        reasons.append("PC arm has alpha <= 0 (would be plain PC)")
    fa, fb = _flat(c_pc), _flat(c_bp)
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k) and k != "arm")
    if diff:
        reasons.append(f"pair configs differ beyond `arm`: {diff}")
    return not reasons, reasons, {"pc": i_pc, "bp": i_bp}


if __name__ == "__main__":
    ok, reasons, info = gate(sys.argv[1], sys.argv[2])
    print(json.dumps({"pass": ok, "reasons": reasons, **info}, indent=2))
    sys.exit(0 if ok else 1)
