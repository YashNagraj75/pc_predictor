#!/usr/bin/env bash
# The two best PC predictors and their frozen BP references on another dataset (plan_datasets_precision.md, rev 3).
#   bash job_datasets.sh <project_root> <job_dir> <mnist|cifar10> <gates|prep|train|eval|all> [smoke]
# Everything for dataset X lives in Q1-PhaseA/ds/X (ds/X_smoke for smoke runs), with the same layout as Q1-PhaseA so the
# Galaxy scripts run unchanged from that directory.  Galaxy10 settings everywhere; training sets matched to Galaxy's
# 15,675 images (config.data_preset).  Nothing is re-tuned.  Rev 3 applies the 27 findings of three reviewers.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); DS="$3"; STAGE="$4"; SMOKE="${5:-}"
cd "$P/Q1-PhaseA" || exit 2
Q=$(pwd); PY="$(cd ../.venv/bin && pwd)/python"            # venv path NOT resolved (review P4)
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=$Q/datasets_status.txt
note() { echo "$(date +%m-%d_%H:%M) [$DS/$STAGE${SMOKE:+/smoke}] $*" | tee -a "$ST"; }
case "$DS" in mnist|cifar10) ;; *) echo "bad dataset $DS"; exit 2 ;; esac
PAR=${PAR:-3}
note "start at $(git rev-parse --short HEAD); $($PY -c 'import sys, jax; print(sys.prefix, jax.__version__)' 2>&1 | tail -1)"

# ---------------------------------------------------------------- data (idempotent, checksummed; review P9)
mkdir -p ../data
MD5_MNIST="f68b3c2dcbeaaa9fbdd348bbdeb94873  train-images-idx3-ubyte.gz
d53e105ee54ea40749a09fcbcd1e9432  train-labels-idx1-ubyte.gz
9fb629c4189551a2d022fa330f9573f3  t10k-images-idx3-ubyte.gz
ec29112dd5afa0611ce80d1b7f02629c  t10k-labels-idx1-ubyte.gz"
mnist_ok() { ( cd ../data/MNIST/raw 2>/dev/null && echo "$MD5_MNIST" | md5sum -c --quiet - >/dev/null 2>&1 ); }
if [ ! -f ../data/MNIST/raw/.ok ]; then                     # files already in place (earlier job) or from the job dir
  mnist_ok || { mkdir -p ../data/MNIST/raw && cp "$JOBDIR"/*-idx?-ubyte.gz ../data/MNIST/raw/ 2>/dev/null; }
  mnist_ok && touch ../data/MNIST/raw/.ok || { note "MNIST copy or checksum failed"; exit 3; }
fi
# an earlier job (d857b6b) downloaded and md5-checked the archive before extracting in place; the loader gates
# (full 50,000 / 10,000 counts) catch a partial extraction
[ -f ../data/cifar10/cifar-10-batches-py/test_batch ] && [ ! -f ../data/cifar10/c10.tgz ] && touch ../data/cifar10/.ok
if [ ! -f ../data/cifar10/.ok ]; then
  T=$(mktemp -d ../data/.c10_XXXX) && ( cd "$T" && curl -sSL -o c10.tgz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
    && echo "c58f30108f718f92721af3b95e74349a  c10.tgz" | md5sum -c - && tar xzf c10.tgz && rm c10.tgz ) \
    && mkdir -p ../data/cifar10 && rm -rf ../data/cifar10/cifar-10-batches-py && mv "$T/cifar-10-batches-py" ../data/cifar10/ \
    && rm -rf "$T" && touch ../data/cifar10/.ok || { note "CIFAR-10 download failed"; rm -rf "$T"; exit 3; }
fi
mkdir -p ds; [ -e ds/data ] || ln -s ../../data ds/data

# ---------------------------------------------------------------- gates (Q1-PhaseA; Galaxy regression and calibration)
if [ "$STAGE" = gates ] || [ "$STAGE" = all ]; then
  $PY -W ignore test_datasets.py > "$JOBDIR/gates.log" 2>&1; rc=$?
  note "gates rc=$rc $(grep -c PASS "$JOBDIR/gates.log") pass $(grep -c FAIL "$JOBDIR/gates.log") fail"; [ $rc -eq 0 ] || exit 4
  $PY -W ignore ds_gate.py --run runs/bp_sigreg_etafix_s0 --out "$JOBDIR/backbone_gate_galaxy.json" --report-only > "$JOBDIR/backbone_gate_galaxy.log" 2>&1
  note "Galaxy backbone gate (calibration): $(tail -1 "$JOBDIR/backbone_gate_galaxy.log" | cut -c1-260)"
fi

W=ds/$DS; [ -n "$SMOKE" ] && W=ds/${DS}_smoke
mkdir -p "$W/pred_runs" "$W/logs" && cd "$W" || exit 2
exec 9>"$Q/ds/.lock_${DS}${SMOKE:+_smoke}"                    # one job per dataset dir; orphaned children keep it (NF3)
flock -n 9 || { note "another job (or an orphaned process) holds the lock for $W"; exit 10; }
collect() {                                    # results + weight manifest back to the job dir (review P6)
  ( find . -name '*.eqx' -o -name 'pred_latents.npz' | sort | xargs -r sha256sum > MANIFEST_weights.txt
    cd "$Q" && tar czf "$JOBDIR/ds_${DS}${SMOKE:+_smoke}.tar.gz" --exclude='*.eqx' --exclude='pred_latents*.npz' "$W" ) \
    || note "collect tar failed"
}
trap collect EXIT
if [ -n "$SMOKE" ]; then ENC=200; NTR=2048; GST=30; FST=30; EV="--smoke"; SEEDS="0"; else ENC=17200; NTR=-1; GST=4000; FST=6000; EV=""; SEEDS="0 1 2"; fi

# ---------------------------------------------------------------- prep: frozen backbone (feature extractor only) + latents
if [ "$STAGE" = prep ] || [ "$STAGE" = all ]; then
  if [ ! -f runs/bp_sigreg_enc_s0/.complete ]; then
    rm -f pred_latents.npz pred_latents.backbone_sha256        # latents must never outlive their backbone (review P7)
    [ -d pred_runs/tests ] && mv pred_runs "old_pred_runs_$(date +%m%d_%H%M)" && mkdir -p pred_runs   # nor predictors (NF4)
    rm -rf runs/bp_sigreg_enc_s0
    $PY -W ignore $Q/run_arms.py --dataset $DS --arms bp --seeds 0 --tag enc --steps $ENC --out runs > logs/prep_backbone.log 2>&1
    [ -f runs/bp_sigreg_enc_s0/DIVERGED ] && { note "backbone DIVERGED"; exit 5; }
    [ -f runs/bp_sigreg_enc_s0/params.eqx ] || { note "backbone training failed (no params.eqx)"; exit 5; }
    echo "$ENC" > runs/bp_sigreg_enc_s0/.complete
  fi
  G=""; [ -n "$SMOKE" ] && G="--report-only"                 # smoke backbone is undertrained: report only
  $PY -W ignore $Q/ds_gate.py --run runs/bp_sigreg_enc_s0 --out backbone_gate.json --rel ${REL:-0.02} $G > logs/prep_gate.log 2>&1; rc=$?
  note "backbone gate rc=$rc $(tail -1 logs/prep_gate.log | cut -c1-300)"; [ $rc -eq 0 ] || exit 6
  if [ ! -f pred_latents.npz ]; then
    $PY -W ignore $Q/pred_data.py --run runs/bp_sigreg_enc_s0 --out pred_latents.npz --n-train $NTR > logs/prep_latents.log 2>&1 \
      || { note "latent extraction failed"; exit 7; }
  fi
  $PY -W ignore $Q/ds_check_latents.py --run runs/bp_sigreg_enc_s0 > logs/prep_check.log 2>&1; rc=$?
  note "latent check rc=$rc $(tail -1 logs/prep_check.log | cut -c1-200)"; [ $rc -eq 0 ] || exit 7
  cp $Q/pred_runs/gpc_selected.json pred_runs/gpc_selected.json && $PY -c "
import json; s = json.load(open('pred_runs/gpc_selected.json'))
for k in ('gpc_pc_d10_zero', 'gpc_bp_d10_zero'): assert isinstance(s[k]['c'], (int, float)), k
print('gsel', {k: s[k]['c'] for k in ('gpc_pc_d10_zero', 'gpc_bp_d10_zero')})" >> logs/prep_check.log 2>&1 \
    || { note "Galaxy gpc_selected.json missing or lacks the d10 keys"; exit 7; }   # Galaxy step scales, unchanged (user)
  note "prep done: $(du -h pred_latents.npz | cut -f1) latents"
fi

# ---------------------------------------------------------------- train: G, F and the two frozen BP references
# Resume skips a run only when its final outputs exist (eqx for F/BPf, json for G/BPg).  Longest runs first (review N4).
if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
  mkdir -p pred_runs/tests_sep_fix pred_runs/tests pred_runs/ffres_fix pred_runs/frozen_bp_fix pred_runs/tests_ffres_fix/R16 pred_runs/tests_ffres_bpfix
  GJ=pred_runs/tests_sep_fix/gpc_sep_d10_amort_hp16_a1_k1_amort_c1_T2200_bp_s; BJ=pred_runs/tests/gpc_bp_d10_zero_bp_s
  FM=pred_runs/ffres_fix/ffres_almR16_d10_s; BM=pred_runs/frozen_bp_fix/ffres_bp_d10_s
  { for s in $SEEDS; do
      echo "G_s$s|[ -f $GJ$s.json ] || $PY -W ignore $Q/run_gpc_tests.py --rule sep --backbone bp --seed $s --depth 10 --belief amort --start amort --rho 16 --k-in 1 --T2 200 --c 1 --alpha 1 --ckpt --steps $GST $EV --outdir pred_runs/tests_sep_fix"
      echo "BPg_s$s|[ -f $BJ$s.json ] || $PY -W ignore $Q/run_gpc_tests.py --rule bp --backbone bp --seed $s --depth 10 --start zero --steps $GST $EV --outdir pred_runs/tests"
    done
    for s in $SEEDS; do
      echo "F_s$s|{ [ -f $FM$s.eqx ] || { rm -f $FM$s.json; $PY -W ignore $Q/pred_ffres.py --arm alm --depth 10 --seed $s --T 400 --alpha 1.0 --lr 0.01 --rho 16 --steps $FST --save --out $FM$s.json; }; } && { [ -f pred_runs/tests_ffres_fix/R16/ffres_alm_d10_bp_s$s.json ] || $PY -W ignore $Q/run_pred_tests.py ffres --arm alm --depth 10 --seed $s --model $FM$s.eqx --outdir pred_runs/tests_ffres_fix/R16; }"
      echo "BPf_s$s|{ [ -f $BM$s.eqx ] || { rm -f $BM$s.json; $PY -W ignore $Q/pred_ffres.py --arm bp --depth 10 --seed $s --lr 0.001 --steps $FST --save --out $BM$s.json; }; } && { [ -f pred_runs/tests_ffres_bpfix/ffres_bp_d10_bp_s$s.json ] || $PY -W ignore $Q/run_pred_tests.py ffres --arm bp --depth 10 --seed $s --model $BM$s.eqx --outdir pred_runs/tests_ffres_bpfix; }"
    done; } > logs/runs.txt
  rm -f logs/*.rc                                              # one status file per run, fresh each launch (review P1/P5)
  run_one() { n="${1%%|*}"; cmd="${1#*|}"; bash -c "set -o pipefail; $cmd" > "logs/$n.log" 2>&1; rc=$?
              echo $rc > "logs/$n.rc"; echo "$(date +%m-%d_%H:%M) rc=$rc $n" >> logs/runs_status.txt; }
  export -f run_one
  tr '\n' '\0' < logs/runs.txt | xargs -0 -P $PAR -I{} bash -c 'run_one "$@"' _ {}
  NEXP=$(wc -l < logs/runs.txt); NOK=$(grep -lx 0 logs/*.rc 2>/dev/null | wc -l)
  note "training done: $NOK of $NEXP ok; failed: $(grep -Lx 0 logs/*.rc 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
  [ "$NOK" -eq "$NEXP" ] || exit 8                             # never evaluate an incomplete set (review DS-2)
fi

# ---------------------------------------------------------------- eval: 1-192 whole views, mean + precision pools, kNN
if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
  if [ -n "$SMOKE" ]; then EA="--n 100 --vmax 48 --seeds 0 --diag-seed 0"; else EA=""; fi
  $PY -W ignore $Q/diag_saccade2.py --whole-only --depths 10 --no-diag --require-all --backbone runs/bp_sigreg_enc_s0 \
      --out pred_runs/eval_views.json $EA > logs/eval.log 2>&1; rc=$?
  note "eval rc=$rc"; [ $rc -eq 0 ] || exit 9
fi
note "DONE"
