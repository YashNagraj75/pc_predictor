#!/usr/bin/env bash
# Seeds 0-2 for the three amortisation arms of E2/E3 (rule sep), depths 3 and 10, with the GPU numerics fix.
# Settings frozen from pred_runs/sep_sel/selected.json (PC-ALM rho 16, k_in 1, T2 200; alpha-0 controls rho 1).
# Seed 0 is rerun because the original screen trained without the fix.  BP references are frozen; none retrained.
#   bash job_sep_seeds.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0      # compiled-GPU gradient fix (2026-10-25)
PAR=${PAR:-3}; ST=sepseeds_status.txt; OUT=pred_runs/tests_sep_fix; LG=pred_runs/logs_sepseeds
note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_sepseeds.tar.gz" --exclude='*.eqx' $ST $OUT $LG sepseeds_audit.json 2>/dev/null || true; }
trap collect EXIT
mkdir -p $OUT $LG
note "start at $(git rev-parse --short HEAD), env TF32=$NVIDIA_TF32_OVERRIDE XLA_FLAGS=$XLA_FLAGS"
# gate: compiled == uncompiled for the sep paths under the fix
$PY -W ignore audit_jit.py sepseeds_audit.json sep > $LG/audit.log 2>&1 || { note "AUDIT FAILED TO RUN"; tail -5 $LG/audit.log; exit 4; }
python3 - <<'PYEOF' || { note "AUDIT GATE FAILED: compiled != uncompiled under the fix"; exit 4; }
import json, sys
c = json.load(open("sepseeds_audit.json"))["cases"]
bad = {k: v for k, v in c.items() if "error" in v or v["cos"] < 0.999}
print({k: round(v.get("cos", -1), 5) for k, v in c.items()})
sys.exit(1 if bad else 0)
PYEOF
note "audit gate passed: $(python3 -c "import json;print({k:round(v['cos'],5) for k,v in json.load(open('sepseeds_audit.json'))['cases'].items()})")"
G="$PY -W ignore run_gpc_tests.py --rule sep --backbone bp"
{ for s in 0 1 2; do for d in 10 3; do
    echo "$G --seed $s --depth $d --belief amort --start amort --rho 16 --k-in 1 --T2 200 --c 1 --alpha 1 --ckpt --outdir $OUT > $LG/amort_alm_d${d}_s$s.log 2>&1; rc=\$?; echo \$(date +%m-%d_%H:%M) rc=\$rc amort_alm_d${d}_s$s >> $ST"
    echo "$G --seed $s --depth $d --belief amort --start amort --rho 1 --k-in 1 --T2 200 --c 1 --alpha 0 --ckpt --outdir $OUT > $LG/amort_none_d${d}_s$s.log 2>&1; rc=\$?; echo \$(date +%m-%d_%H:%M) rc=\$rc amort_none_d${d}_s$s >> $ST"
    echo "$G --seed $s --depth $d --belief pc --start zero --rho 1 --k-in 1 --T2 200 --c 1 --alpha 0 --ckpt --outdir $OUT > $LG/pc_none_d${d}_s$s.log 2>&1; rc=\$?; echo \$(date +%m-%d_%H:%M) rc=\$rc pc_none_d${d}_s$s >> $ST"
  done; done; } > $LG/runs.txt
# resume: drop runs already finished (rc=0 recorded in the status file)
python3 - "$ST" "$LG/runs.txt" <<'PYEOF'
import re, sys
done = set(re.findall(r"rc=0 (\S+)", open(sys.argv[1]).read()))
lines = [l for l in open(sys.argv[2]) if re.search(r"rc=\$rc (\S+) >>", l).group(1) not in done]
open(sys.argv[2], "w").writelines(lines); print(f"runs to do: {len(lines)} (skipping {len(done)} done)")
PYEOF
note "resume: $(wc -l < $LG/runs.txt) runs to do"
xargs -d '\n' -P $PAR -I{} bash -c '{}' < $LG/runs.txt
note "ALL DONE: $(ls $OUT/gpc_sep_*_s?.json 2>/dev/null | wc -l) results, $(ls $OUT/*.diverged.json 2>/dev/null | wc -l) failed"
exit 0
