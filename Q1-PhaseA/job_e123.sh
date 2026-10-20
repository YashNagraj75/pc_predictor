#!/usr/bin/env bash
# E1 residual feedforward PC-ALM (d3, d10) + E2/E3 generative belief/credit separation (d3, d10).
# Spec: wiki 'Method - Deep feedforward PC-ALM and belief-credit separation plan'; prereg ffres_prereg.json, sep_prereg.json.
# Resumable: every run skips an existing output.  Frozen BP references are never retrained.
#   bash job_e123.sh <project_root> <job_dir> [e1|e23|all]
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); WHAT="${3:-all}"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
PAR=${PAR:-3}; ST=e123_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
FZ=pred_runs/frozen_bp; FR=pred_runs/ffres; FS=pred_runs/ffres_sel; SS=pred_runs/sep_sel; TS=pred_runs/tests_sep; LG=pred_runs/logs_e123
collect() { tar czf "$JOBDIR/q1_e123.tar.gz" --exclude='*.eqx' $ST e123_gates_*.log ffres_prereg.json sep_prereg.json \
            $FZ $FR $FS $SS $TS $LG pred_runs/tests_ffres 2>/dev/null || true; }
trap collect EXIT
mkdir -p $FZ $FR $FS $SS $TS $LG pred_runs/tests_ffres
note "start at $(git rev-parse --short HEAD) (dirty tracked files: $(git status --porcelain --untracked-files=no | wc -l)), part=$WHAT"
pool() { xargs -d '\n' -P $PAR -I{} bash -c '{}' < "$1"; collect; }
# each generated line runs one command, then records ITS exit code (review N8)
ff() { echo "$PY -W ignore pred_ffres.py $1 > $LG/$(basename $2 .json).log 2>&1; rc=\$?; echo \$(date +%m-%d_%H:%M) rc=\$rc $2 >> $ST"; }
G="$PY -W ignore run_gpc_tests.py --rule sep --backbone bp --seed 0"
gp() { t=$(echo "$1" | tr -d '-' | tr ' /' '__' | cut -c1-110); echo "$G $1 > $LG/$t.log 2>&1; rc=\$?; echo \$(date +%m-%d_%H:%M) rc=\$rc gpc $t >> $ST"; }
jget() { python3 -c "import json,sys; d=json.load(open('$1')); print(eval('d' + sys.argv[1]))" "$2"; }

# ---------------------------------------------------------------- 1. gates (one depth per process) + key check
python3 -c "import json; d=json.load(open('pred_runs/gpc_selected.json')); [d[f'gpc_{r}_d{x}_zero']['c'] for r in ('pc','bp') for x in (3,10)]" \
  || { note "gpc_selected.json lacks a stage-1 key"; exit 4; }
for d in 3 10; do
  $PY -W ignore test_ffres.py $d > e123_gates_ffres_d$d.log 2>&1 || { note "GATES FAILED ffres d$d"; grep -h FAIL e123_gates_ffres_d$d.log | head; exit 4; }
  $PY -W ignore test_sep.py $d > e123_gates_sep_d$d.log 2>&1 || { note "GATES FAILED sep d$d"; grep -h FAIL e123_gates_sep_d$d.log | head; exit 4; }
  $PY -W ignore test_sep.py $d --x64 > e123_gates_sep64_d$d.log 2>&1 || { note "GATES FAILED sep64 d$d"; grep -h FAIL e123_gates_sep64_d$d.log | head; exit 4; }
done
note "gates pass: $(cat e123_gates_*.log | grep -c PASS)"

# ---------------------------------------------------------------- 2. smoke: every code path once, both depths (fresh each launch)
rm -rf $LG/smoke_sep $LG/smoke_ffres_*
for d in 3 10; do
  for arm in bp pc alm; do
    $PY -W ignore pred_ffres.py --arm $arm --depth $d --steps 30 --T 5 --save --out $LG/smoke_ffres_${arm}_d$d.json > $LG/smoke_ffres_${arm}_d$d.log 2>&1 \
      && $PY -W ignore run_pred_tests.py ffres --arm $arm --depth $d --model $LG/smoke_ffres_${arm}_d$d.eqx --outdir $LG/smoke_ffres_tests >> $LG/smoke_ffres_${arm}_d$d.log 2>&1 \
      || { note "SMOKE FAILED ffres $arm d$d"; tail -5 $LG/smoke_ffres_${arm}_d$d.log; exit 5; }
  done
  for b in oracle pc amort; do st=zero; [ $b = amort ] && st=amort
    for al in 1 0; do
      $PY -W ignore run_gpc_tests.py --rule sep --belief $b --start $st --depth $d --rho 16 --k-in 5 --alpha $al --T 10 --T2 10 --steps 8 \
          --ckpt --smoke --outdir $LG/smoke_sep > $LG/smoke_sep_${b}_a${al}_d$d.log 2>&1 || { note "SMOKE FAILED sep $b a$al d$d"; tail -5 $LG/smoke_sep_${b}_a${al}_d$d.log; exit 5; }
    done
  done
  $PY -W ignore run_gpc_tests.py --rule sep --belief oracle --start zero --depth $d --rho 16 --k-in 5 --T 10 --T2 10 --steps 8 --holdout \
      --outdir $LG/smoke_sep/holdout > $LG/smoke_sep_holdout_d$d.log 2>&1 || { note "SMOKE FAILED sep holdout d$d"; exit 5; }
done
note "smoke ok"

# ---------------------------------------------------------------- 3. frozen generative BP reference: manifest written once, then verified
$PY - <<'PYEOF' || { note "generative BP manifest mismatch"; exit 4; }
import glob, hashlib, json, os
fs = sorted(glob.glob("pred_runs/tests/gpc_bp_d*_zero_bp_s[012].eqx"))
if len(fs) != 6:
    print(f"WARNING: expected 6 frozen BP auto-decoders, found {len(fs)}: {fs}")
m = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in fs}
path = "pred_runs/frozen_bp/MANIFEST_generative.json"
if os.path.exists(path):
    old = json.load(open(path))["files"]
    assert old == m, "a frozen generative BP reference changed"
else:
    json.dump({"what": "frozen generative BP auto-decoder references (G1); never retrained",
               "test_k1_nmse_3seed_mean": {"d3": 0.3053, "d10": 0.2783}, "files": m}, open(path, "w"), indent=1)
PYEOF
note "generative BP references verified: $(grep -c '.eqx' pred_runs/frozen_bp/MANIFEST_generative.json) files"

# ---------------------------------------------------------------- 4. E1 residual feedforward (muPC network for every rule)
# BP_LR=select lets the frozen BP reference choose its learning rate once from the same grid (user decision pending);
# otherwise BP trains at lr 1e-3, as every earlier feedforward predictor.
if [ "$WHAT" = all ] || [ "$WHAT" = e1 ]; then
LRS="0.001 0.003 0.01"; BP_LR=${BP_LR:-0.001}
{ for d in 3 10; do for lr in $LRS; do
    for T in 20 100; do
      for A in 0.25 1.0; do o=$FS/alm_d${d}_T${T}_a${A}_lr$lr.json; [ -f $o ] || ff "--arm alm --depth $d --T $T --alpha $A --lr $lr --holdout --out $o" $o; done
      o=$FS/pc_d${d}_T${T}_lr$lr.json; [ -f $o ] || ff "--arm pc --depth $d --T $T --lr $lr --holdout --out $o" $o
    done
    if [ "$BP_LR" = select ]; then o=$FS/bp_d${d}_lr$lr.json; [ -f $o ] || ff "--arm bp --depth $d --lr $lr --holdout --out $o" $o; fi
  done; done
} > $LG/e1_sel.txt
pool $LG/e1_sel.txt
$PY - <<'PYEOF'
import json, glob, math
out = {"rule": "per depth and arm: lowest held-out k=1 nmse, seed 0, 90/10 train split; PC-ALM alpha {0.25,1} x T {20,100} x lr {1e-3,3e-3,1e-2}; plain PC T {20,100} x lr; BP lr only if BP_LR=select"}
for d in (3, 10):
    for arm in ("alm", "pc", "bp"):
        rows = [json.load(open(f)) for f in glob.glob(f"pred_runs/ffres_sel/{arm}_d{d}_*.json")]
        if not rows:
            continue
        tab = [{"T": r["cfg"]["T"], "alpha": r["cfg"]["alpha"], "lr": r["cfg"]["lr"], "val_nmse": r["test"]["nmse"], "finite": r["finite"]} for r in rows]
        ok = [t for t in tab if t["finite"] and t["val_nmse"] is not None and math.isfinite(t["val_nmse"])]
        b = min(ok, key=lambda t: t["val_nmse"]) if ok else {"T": 20, "alpha": 1.0, "lr": 1e-3, "val_nmse": None}
        out[f"{arm}_d{d}"] = {"T": b["T"], "alpha": b["alpha"], "lr": b["lr"], "val_nmse": b["val_nmse"], "selection_failed": not ok, "all": tab}
json.dump(out, open("pred_runs/ffres_sel/selected.json", "w"), indent=1)
PYEOF
note "E1 selected: $(python3 -c "import json;d=json.load(open('$FS/selected.json'));print({k:(v['T'],v['alpha'],v['lr'],v['selection_failed']) for k,v in d.items() if k!='rule'})")"
{ for d in 3 10; do
    TA=$(jget $FS/selected.json "['alm_d$d']['T']"); AA=$(jget $FS/selected.json "['alm_d$d']['alpha']"); LA=$(jget $FS/selected.json "['alm_d$d']['lr']")
    TP=$(jget $FS/selected.json "['pc_d$d']['T']"); LP=$(jget $FS/selected.json "['pc_d$d']['lr']")
    LB=$BP_LR; [ "$BP_LR" = select ] && LB=$(jget $FS/selected.json "['bp_d$d']['lr']")
    for s in 0 1 2; do
      o=$FZ/ffres_bp_d${d}_s$s.json;  [ -f $o ] || ff "--arm bp --depth $d --seed $s --lr $LB --save --out $o" $o
      o=$FR/ffres_alm_d${d}_s$s.json; [ -f $o ] || ff "--arm alm --depth $d --seed $s --T $TA --alpha $AA --lr $LA --save --out $o" $o
      o=$FR/ffres_pc_d${d}_s$s.json;  [ -f $o ] || ff "--arm pc --depth $d --seed $s --T $TP --lr $LP --save --out $o" $o
    done; done; } > $LG/e1_runs.txt
pool $LG/e1_runs.txt
python3 - <<'PYEOF'
import json, os
fz, m = "pred_runs/frozen_bp", {}
for d in (3, 10):
    for s in (0, 1, 2):
        j = f"{fz}/ffres_bp_d{d}_s{s}.json"
        if os.path.exists(j):
            r = json.load(open(j))
            m[j] = {k: r.get(k) for k in ("eqx_sha256", "commit", "dirty_files", "matmul_precision")} | {"lr": r["cfg"]["lr"], "test_nmse": r["test"]["nmse"]}
path = f"{fz}/MANIFEST_ffres.json"
if not os.path.exists(path):
    json.dump({"what": "frozen residual feedforward BP references (muPC network, exact gradient); never retrained", "runs": m},
              open(path, "w"), indent=1)
PYEOF
note "E1 trained; frozen ffres BP manifest present"
for d in 3 10; do for s in 0 1 2; do
  for pair in "bp:$FZ" "alm:$FR" "pc:$FR"; do arm=${pair%%:*}; dir=${pair#*:}; m=$dir/ffres_${arm}_d${d}_s$s.eqx
    [ -f pred_runs/tests_ffres/ffres_${arm}_d${d}_bp_s$s.json ] || [ ! -f $m ] || \
      $PY -W ignore run_pred_tests.py ffres --arm $arm --depth $d --seed $s --model $m --outdir pred_runs/tests_ffres > $LG/test_ffres_${arm}_d${d}_s$s.log 2>&1 \
      || note "TEST FAILED ffres $arm d$d s$s"
  done; done; done
collect
note "E1 tests done: $(ls pred_runs/tests_ffres/*.json 2>/dev/null | wc -l)"
fi

# ---------------------------------------------------------------- 5. E2/E3 selection
if [ "$WHAT" = all ] || [ "$WHAT" = e23 ]; then
# 5a. A1 (oracle belief + PC-ALM), rho 16, k_in {1,5} x T2 {200,800}, c = 1, no retries (a failure is a result)
{ for d in 10 3; do for k in 1 5; do for T2 in 200 800; do
    gp "--depth $d --belief oracle --start zero --rho 16 --k-in $k --T2 $T2 --c 1 --alpha 1 --holdout --outdir $SS"; done; done; done; } > $LG/sep_sel_a1.txt
pool $LG/sep_sel_a1.txt
pick() {   # $1 = depth, $2 = glob tag, $3 = output key; falls back to the least-bad failed run (review N3)
$PY - "$1" "$2" "$3" <<'PYEOF'
import json, glob, math, os, sys
d, tag, key = sys.argv[1:4]
path = "pred_runs/sep_sel/selected.json"
out = json.load(open(path)) if os.path.exists(path) else {}
rows = []
for f in glob.glob(f"pred_runs/sep_sel/gpc_sep_d{d}_*{tag}*.json"):
    r = json.load(open(f)); cfg = r["cfg"]; fail = f.endswith(".diverged.json")
    n = (r.get("rows_last_finite") or [{}])[0].get("nmse") if fail else r["val_k1"]["nmse"]
    rows.append({"rho": cfg["hid_prec"], "k_in": cfg["k_inner"], "T2": cfg["T2"], "failed": fail,
                 "val_nmse": n if (n is not None and math.isfinite(n)) else None})
ok = [r for r in rows if not r["failed"] and r["val_nmse"] is not None]
cand = ok or [r for r in rows if r["val_nmse"] is not None] or rows
b = min(cand, key=lambda r: r["val_nmse"] if r["val_nmse"] is not None else 1e9)
out[key] = {**b, "selection_failed": not ok, "all": rows}
json.dump(out, open(path, "w"), indent=1)
print(key, b["rho"], b["k_in"], b["T2"], "FAILED" if not ok else "")
PYEOF
}
for d in 3 10; do pick $d "_a1_" "A1_d$d" >> $ST; done
# 5b. alpha = 0 controls choose their own hidden precision rho {1,16} at A1's T2 (review F5)
{ for d in 10 3; do T2=$(jget $SS/selected.json "['A1_d$d']['T2']")
    for r in 1 16; do gp "--depth $d --belief oracle --start zero --rho $r --k-in 1 --T2 $T2 --c 1 --alpha 0 --holdout --outdir $SS"; done; done; } > $LG/sep_sel_a0.txt
pool $LG/sep_sel_a0.txt
for d in 3 10; do pick $d "_a0_" "A0_d$d" >> $ST; done
note "sep selected: $(python3 -c "import json;d=json.load(open('$SS/selected.json'));print({k:(v['rho'],v['k_in'],v['T2'],v['selection_failed']) for k,v in d.items()})")"

# ---------------------------------------------------------------- 6. E2/E3 screen: belief x {PC-ALM, alpha 0}, seed 0
{ for d in 10 3; do
    R1=$(jget $SS/selected.json "['A1_d$d']['rho']"); K1=$(jget $SS/selected.json "['A1_d$d']['k_in']"); T2=$(jget $SS/selected.json "['A1_d$d']['T2']")
    R0=$(jget $SS/selected.json "['A0_d$d']['rho']")
    for b in oracle pc amort; do st=zero; [ $b = amort ] && st=amort
      gp "--depth $d --belief $b --start $st --rho $R1 --k-in $K1 --T2 $T2 --c 1 --alpha 1 --ckpt --outdir $TS"
      gp "--depth $d --belief $b --start $st --rho $R0 --k-in 1 --T2 $T2 --c 1 --alpha 0 --ckpt --outdir $TS"
    done; done; } > $LG/sep_screen.txt
pool $LG/sep_screen.txt
note "E2/E3 done: $(ls $TS/gpc_sep_*_s0.json 2>/dev/null | grep -vc diverged) results, $(ls $TS/*.diverged.json 2>/dev/null | wc -l) failed"
fi
note "ALL DONE ($WHAT)"
