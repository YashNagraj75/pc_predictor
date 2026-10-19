#!/usr/bin/env bash
# E1 residual feedforward PC-ALM (d3, d10) + E2/E3 generative belief/credit separation (d3, d10).
# Spec: wiki 'Method - Deep feedforward PC-ALM and belief-credit separation plan'; prereg ffres_prereg.json, sep_prereg.json.
# Resumable: every run skips an existing output.  Frozen BP references are never retrained.
#   bash job_e123.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR="$2"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PAR=${PAR:-3}; ST=e123_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
FZ=pred_runs/frozen_bp; FR=pred_runs/ffres; FS=pred_runs/ffres_sel; SS=pred_runs/sep_sel; TS=pred_runs/tests_sep; LG=pred_runs/logs_e123
collect() { tar czf "$JOBDIR/q1_e123.tar.gz" --exclude='*.eqx' $ST e123_gates_*.log ffres_prereg.json sep_prereg.json \
            $FZ $FR $FS $SS $TS $LG pred_runs/tests_ffres 2>/dev/null || true; }
trap collect EXIT
mkdir -p $FZ $FR $FS $SS $TS $LG pred_runs/tests_ffres
note "start at $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- 1. gates (GPU; one depth per process)
for d in 3 10; do
  $PY -W ignore test_ffres.py $d > e123_gates_ffres_d$d.log 2>&1 || { note "GATES FAILED ffres d$d"; grep -h FAIL e123_gates_ffres_d$d.log | head; exit 4; }
  $PY -W ignore test_sep.py $d > e123_gates_sep_d$d.log 2>&1 || { note "GATES FAILED sep d$d"; grep -h FAIL e123_gates_sep_d$d.log | head; exit 4; }
  $PY -W ignore test_sep.py $d --x64 > e123_gates_sep64_d$d.log 2>&1 || { note "GATES FAILED sep64 d$d"; grep -h FAIL e123_gates_sep64_d$d.log | head; exit 4; }
done
note "gates pass: $(cat e123_gates_*.log | grep -c PASS)"

# ---------------------------------------------------------------- 2. smoke (every code path once)
for arm in bp pc alm; do
  $PY -W ignore pred_ffres.py --arm $arm --depth 10 --steps 30 --T 5 --out $LG/smoke_ffres_$arm.json > $LG/smoke_ffres_$arm.log 2>&1 \
    || { note "SMOKE FAILED ffres $arm"; tail -5 $LG/smoke_ffres_$arm.log; exit 5; }
done
for b in oracle pc amort; do
  st=zero; [ $b = amort ] && st=amort
  $PY -W ignore run_gpc_tests.py --rule sep --belief $b --start $st --depth 10 --rho 4 --k-in 5 --T2 10 --steps 8 --holdout \
      --outdir $LG/smoke_sep > $LG/smoke_sep_$b.log 2>&1 || { note "SMOKE FAILED sep $b"; tail -5 $LG/smoke_sep_$b.log; exit 5; }
done
note "smoke ok"

# ---------------------------------------------------------------- 3. freeze the generative BP reference (manifest only)
$PY - <<'PYEOF'
import glob, hashlib, json, os
fs = sorted(glob.glob("pred_runs/tests/gpc_bp_d*_zero_bp_s[012].eqx"))
m = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in fs}
json.dump({"what": "frozen generative BP auto-decoder references (G1); never retrained", "files": m},
          open("pred_runs/frozen_bp/MANIFEST_generative.json", "w"), indent=1)
print(len(m), "files")
PYEOF
note "generative BP manifest: $(grep -c sha pred_runs/frozen_bp/MANIFEST_generative.json 2>/dev/null || grep -o 'eqx' pred_runs/frozen_bp/MANIFEST_generative.json | wc -l) entries"

pool() { xargs -d '\n' -P $PAR -I{} bash -c '{}' < "$1"; }
ff() { echo "$PY -W ignore pred_ffres.py $1 > $LG/$(basename $2 .json).log 2>&1; echo \$(date +%m-%d_%H:%M) rc=\$? $2 >> $ST"; }
export PY ST LG

# ---------------------------------------------------------------- 4. E1 residual feedforward
{ for d in 3 10; do for s in 0 1 2; do o=$FZ/ffres_bp_d${d}_s$s.json; [ -f $o ] || ff "--arm bp --depth $d --seed $s --save --out $o" $o; done; done
  for T in 20 100; do for A in 0.25 1.0; do o=$FS/alm_d10_T${T}_a${A}.json; [ -f $o ] || ff "--arm alm --depth 10 --T $T --alpha $A --holdout --out $o" $o; done; done
} > $LG/e1_wave1.txt
pool $LG/e1_wave1.txt
$PY - <<'PYEOF' || { note "E1 selection failed"; exit 6; }
import json, glob, math
rows = [json.load(open(f)) for f in glob.glob("pred_runs/ffres_sel/alm_d10_*.json")]
rows = [r for r in rows if r["finite"] and r["test"]["nmse"] is not None and math.isfinite(r["test"]["nmse"])]
assert rows, "no finite selection run"
b = min(rows, key=lambda r: r["test"]["nmse"])
json.dump({"T": b["cfg"]["T"], "alpha": b["cfg"]["alpha"], "val_nmse": b["test"]["nmse"],
           "all": [{"T": r["cfg"]["T"], "alpha": r["cfg"]["alpha"], "val_nmse": r["test"]["nmse"]} for r in rows],
           "rule": "lowest held-out k=1 nmse, d10, seed 0, alpha {0.25,1} x T {20,100}; used at both depths"},
          open("pred_runs/ffres_sel/selected.json", "w"), indent=1)
print(b["cfg"]["T"], b["cfg"]["alpha"])
PYEOF
T=$(python3 -c "import json;print(json.load(open('$FS/selected.json'))['T'])"); A=$(python3 -c "import json;print(json.load(open('$FS/selected.json'))['alpha'])")
note "E1 selected T=$T alpha=$A"
{ for d in 3 10; do for s in 0 1 2; do
    o=$FR/ffres_alm_d${d}_s$s.json; [ -f $o ] || ff "--arm alm --depth $d --seed $s --T $T --alpha $A --save --out $o" $o
    o=$FR/ffres_pc_d${d}_s$s.json;  [ -f $o ] || ff "--arm pc --depth $d --seed $s --T $T --save --out $o" $o
  done; done; } > $LG/e1_wave2.txt
pool $LG/e1_wave2.txt
python3 - <<'PYEOF'
import json, os
fz = "pred_runs/frozen_bp"
m = {}
for d in (3, 10):
    for s in (0, 1, 2):
        j = f"{fz}/ffres_bp_d{d}_s{s}.json"
        if os.path.exists(j):
            r = json.load(open(j)); m[j] = {"eqx_sha256": r.get("eqx_sha256"), "commit": r.get("commit"), "test_nmse": r["test"]["nmse"]}
json.dump({"what": "frozen residual feedforward BP references (standard parameterisation, no muPC); never retrained", "runs": m},
          open(f"{fz}/MANIFEST_ffres.json", "w"), indent=1)
PYEOF
note "E1 trained; frozen BP manifest written"
for d in 3 10; do for s in 0 1 2; do
  for pair in "bp:$FZ" "alm:$FR" "pc:$FR"; do arm=${pair%%:*}; dir=${pair#*:}; m=$dir/ffres_${arm}_d${d}_s$s.eqx
    [ -f pred_runs/tests_ffres/ffres_${arm}_d${d}_bp_s$s.json ] || [ ! -f $m ] || \
      $PY -W ignore run_pred_tests.py ffres --arm $arm --depth $d --seed $s --model $m --outdir pred_runs/tests_ffres > $LG/test_ffres_${arm}_d${d}_s$s.log 2>&1 \
      || note "TEST FAILED ffres $arm d$d s$s"
  done; done; done
note "E1 tests done: $(ls pred_runs/tests_ffres/*.json 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- 5. E2/E3 selection: oracle belief + PC-ALM, rho x k_in
G="$PY -W ignore run_gpc_tests.py --rule sep --backbone bp --seed 0 --T2 200"
gp() { t=$(echo "$1" | tr -d '-' | tr ' /' '__' | cut -c1-110); echo "$G $1 > $LG/$t.log 2>&1; echo \$(date +%m-%d_%H:%M) rc=\$? gpc $t >> $ST"; }
{ for d in 10 3; do for rho in 4 16; do for k in 1 5; do
    gp "--depth $d --belief oracle --start zero --rho $rho --k-in $k --alpha 1 --holdout --outdir $SS"; done; done; done; } > $LG/sep_sel.txt
pool $LG/sep_sel.txt
$PY - <<'PYEOF' || { note "sep selection failed"; exit 7; }
import json, glob, math
out = {}
for d in (3, 10):
    rows = []
    for f in glob.glob(f"pred_runs/sep_sel/gpc_sep_d{d}_*.json"):
        if f.endswith(".diverged.json"):
            continue
        r = json.load(open(f)); n = r["val_k1"]["nmse"]
        if math.isfinite(n):
            rows.append({"rho": r["cfg"]["hid_prec"], "k_in": r["cfg"]["k_inner"], "c_used": r["c_used"], "val_nmse": n})
    assert rows, f"d{d}: no stable selection run"
    b = min(rows, key=lambda r: r["val_nmse"])
    out[str(d)] = {**b, "all": rows}
out["rule"] = "per depth: lowest held-out k=1 nmse of arm A1 (oracle belief, PC-ALM) over rho {4,16} x k_in {1,5}, full 4000 steps; diverged/exploded excluded; used for every PC arm at that depth"
json.dump(out, open("pred_runs/sep_sel/selected.json", "w"), indent=1)
print({k: (v["rho"], v["k_in"]) for k, v in out.items() if k != "rule"})
PYEOF
note "sep selected: $(python3 -c "import json;d=json.load(open('$SS/selected.json'));print({k:(v['rho'],v['k_in'],v['c_used']) for k,v in d.items() if k!='rule'})")"

# ---------------------------------------------------------------- 6. E2/E3 screen: belief x {PC-ALM, alpha 0}, seed 0
{ for d in 10 3; do
    R=$(python3 -c "import json;print(json.load(open('$SS/selected.json'))['$d']['rho'])")
    K=$(python3 -c "import json;print(json.load(open('$SS/selected.json'))['$d']['k_in'])")
    C=$(python3 -c "import json;print(json.load(open('$SS/selected.json'))['$d']['c_used'])")
    for b in oracle pc amort; do st=zero; [ $b = amort ] && st=amort
      for al in 1 0; do gp "--depth $d --belief $b --start $st --rho $R --k-in $K --c $C --alpha $al --ckpt --outdir $TS"; done; done
  done; } > $LG/sep_screen.txt
pool $LG/sep_screen.txt
note "ALL DONE: $(ls $TS/gpc_sep_*_s0.json 2>/dev/null | grep -vc diverged) results, $(ls $TS/*.diverged.json 2>/dev/null | wc -l) diverged"
