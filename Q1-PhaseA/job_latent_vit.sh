#!/usr/bin/env bash
# Latent analysis job (plan_latent_analysis.md): GPU-vs-CPU gate at both depths, known-answer tests, latent_vit.py on every
# finished stage-A 25-epoch run, then output checks.  bash job_latent_vit.sh <root> <jobdir> <outdir>.  Resumable (skips done runs).
set +e; set -u
P="$1"; JOBDIR=$(readlink -f "$2"); OUT="$3"; cd "$P/Q1-PhaseA" || exit 2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
PY="$(cd ../.venv/bin && pwd)/python"; L="$JOBDIR/latent_sequence.txt"
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$L"; }
for D in 10 6; do bash job_stageA.sh "$P" "$JOBDIR" gpugate $D; rc=$?; log "gpu gate d$D rc=$rc"; [ $rc -eq 0 ] || exit 4; done
$PY -W ignore test_latent_vit.py > "$JOBDIR/test_latent.log" 2>&1; rc=$?; log "tests rc=$rc"; [ $rc -eq 0 ] || exit 5
mkdir -p "$OUT"; $PY -W ignore latent_vit.py "$P" "$OUT" 10 6 > "$JOBDIR/latent.log" 2>&1; rc=$?; log "latent rc=$rc"; [ $rc -eq 0 ] || exit 6
$PY - "$OUT" >> "$L" 2>&1 <<'PYX'
import glob, json, sys
bad = []
for f in sorted(glob.glob(sys.argv[1] + "/latent_*.json")):
    r = json.load(open(f))
    if not r["reproduces_report"]: bad.append((r["run"], "nmse differs from report"))
    for c, v in r["conds"].items():
        if abs(v["band_sum_check"] - 1) > 1e-6 or abs(v["dist_sum_check"] - 1) > 1e-6: bad.append((r["run"], c, v["band_sum_check"], v["dist_sum_check"]))
print("OUTPUT CHECKS", "PASS" if not bad else f"FAIL {bad}")
sys.exit(1 if bad else 0)
PYX
rc=$?; log "output checks rc=$rc"; exit $rc
