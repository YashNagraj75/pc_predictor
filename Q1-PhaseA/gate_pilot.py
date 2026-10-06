"""Pilot gate for a sparse PC arm (rule D5).  Exit 0 = pass, 1 = fail.

    ./.venv/bin/python gate_pilot.py runs/<pilot_run_dir>

Criteria (spec: 'Method - Sparse PC-ALM arms spec'):
  * active fraction of the sparse variable, mean over the last 200 steps, in [0.1, 0.3]
  * gradient clipping fired on < 5 % of all steps
  * everything finite, and act_growth = max_l ||z_l^(T)|| / ||z_l^(0)|| <= 10 over the
    last 200 steps.  (Not F_T / F_0: the completed-square energy grows with the
    multipliers by design.)
A failing arm is skipped and reported -- never re-tuned against the pilot.
"""
import json
import math
import os
import sys

LAST = 200


def gate(run_dir):
    log = json.load(open(os.path.join(run_dir, "log.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    clip = cfg["opt"]["grad_clip"]
    tail = log[-LAST:]
    reasons, info = [], {}

    def col(rows, k):
        return [r[k] for r in rows if k in r]

    af = col(tail, "active_frac")
    # case C clips encoder and dictionary as separate groups: the confound we
    # gate on is the ENCODER being clipped; the dictionary's rate is reported.
    is_C = cfg.get("sparse", {}).get("case") == "C"
    g = col(log, "grad_norm_enc") if is_C else col(log, "grad_norm_preclip")
    if is_C:
        gd = col(log, "grad_norm_dict")
        info["clip_frac_dict"] = sum(v > clip for v in gd) / max(len(gd), 1)
    ag = col(tail, "act_growth")
    lf = col(log, "loss_ff")
    if not af or not g or not ag:
        return False, ["log is missing active_frac / grad_norm_preclip / act_growth"], info
    info["active_frac_last200"] = sum(af) / len(af)
    info["clip_frac"] = sum(v > clip for v in g) / len(g)
    info["act_growth_max_last200"] = max(ag)
    info["n_steps"] = len(log)
    finite = all(math.isfinite(v) for v in g + ag + lf + af)
    if not 0.1 <= info["active_frac_last200"] <= 0.3:
        reasons.append(f"active fraction {info['active_frac_last200']:.3f} outside [0.1, 0.3]")
    if info["clip_frac"] >= 0.05:
        reasons.append(f"clipping fired on {100 * info['clip_frac']:.1f}% of steps (>= 5%)")
    if not finite:
        reasons.append("non-finite values in the log")
    if info["act_growth_max_last200"] > 10:
        reasons.append(f"activity growth {info['act_growth_max_last200']:.2f} > 10")
    return not reasons, reasons, info


if __name__ == "__main__":
    ok, reasons, info = gate(sys.argv[1])
    print(json.dumps({"run": sys.argv[1], "pass": ok, "reasons": reasons, **info}, indent=2))
    sys.exit(0 if ok else 1)
