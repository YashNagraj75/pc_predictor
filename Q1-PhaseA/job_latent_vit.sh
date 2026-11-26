#!/usr/bin/env bash
# Latent analysis (plan_latent_analysis.md, rev 2): GPU-vs-CPU gate at both depths, known-answer tests, then latent_vit.py on
# every finished stage-A 25-epoch run under the GPU lock, then output checks.  bash job_latent_vit.sh <root> <jobdir> <outdir> <pass>
# Resumable: finished runs at the same commit are skipped.  Exit 0 only if everything passed.
set +e; set -u
P="$1"; JOBDIR=$(readlink -f "$2"); OUT="$3"; PASS="$4"; cd "$P/Q1-PhaseA" || exit 2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
PY="$(cd ../.venv/bin && pwd)/python"; L="$JOBDIR/sequence.txt"; B=runs_vit/stageA
log() { echo "$(date '+%m-%d %H:%M') [latent $PASS] $*" >> "$L"; }
for D in 10 6; do bash job_stageA.sh "$P" "$JOBDIR" gpugate $D; rc=$?; log "gpu gate d$D rc=$rc"; [ $rc -eq 0 ] || exit 4; done
$PY -W ignore test_latent_vit.py > "$JOBDIR/test_latent_p$PASS.log" 2>&1; rc=$?; log "tests rc=$rc"; [ $rc -eq 0 ] || exit 5
mkdir -p "$OUT"
flock -n "$B/.gpu.lock" $PY -W ignore latent_vit.py "$P" "$OUT" 10 6 > "$JOBDIR/latent_p$PASS.log" 2>&1; rc=$?; log "latent_vit rc=$rc"; [ $rc -eq 0 ] || exit 6
$PY - "$OUT" "$PASS" >> "$L" 2>&1 <<'PYX'
import glob, json, sys
out, pas = sys.argv[1], int(sys.argv[2]); bad, n = [], {}
for f in sorted(glob.glob(out + "/latent_*.json")):
    r = json.load(open(f)); n[(r["depth"], r["arm"])] = n.get((r["depth"], r["arm"]), 0) + 1
    if r["arm"] != "init" and not r.get("reproduces_report"): bad.append((r["run"], "report", r.get("report_dev")))
    if not (max(r["instr_vs_model_rel"].values()) <= 1e-3): bad.append((r["run"], "instr", r["instr_vs_model_rel"]))
    for c, v in r["conds"].items():
        if not (abs(v["band_sum_check"] - 1) <= 1e-5 and abs(v["dist_sum_check"] - 1) <= 1e-5): bad.append((r["run"], c, v["band_sum_check"], v["dist_sum_check"]))
    if not (r["conds"]["noise"]["noise"]["split_check_rel"] <= 1e-3): bad.append((r["run"], "noise split"))
for d in (6, 10):                                # expected = the run list latent_vit.py wrote for this depth
    man = json.load(open(f"{out}/runs_d{d}.json")); want = {nm for nm, _, _ in man["runs"]}
    have = {json.load(open(f))["run"] for f in glob.glob(out + "/latent_*.json") if json.load(open(f))["depth"] == d}
    if want != have: bad.append((d, "runs vs results", sorted(want - have), sorted(have - want)))
    ar = [a for _, a, _ in man["runs"]]
    for a, k in {"bp": 5, "init": 1, "epc_alm_k4": 1, "epc_k4": 1}.items():
        if ar.count(a) != k: bad.append((d, a, "count", ar.count(a), "expected", k))
    if pas >= 3 and d == 10 and sorted(s for _, a, s in man["runs"] if a == "epc_alm_blk4") != [0, 1, 2]: bad.append((d, "epc_alm_blk4 seeds", "expected 0,1,2"))
    ck = json.load(open(f"{out}/cka_d{d}.json"))["pairs"]; m = sum(v for (dd, _), v in n.items() if dd == d)
    if len(ck) != m * (m - 1) // 2: bad.append((d, "cka pairs", len(ck), m * (m - 1) // 2))
print("OUTPUT CHECKS", "PASS" if not bad else f"FAIL {bad}")
sys.exit(1 if bad else 0)
PYX
rc=$?; log "output checks rc=$rc"; exit $rc
