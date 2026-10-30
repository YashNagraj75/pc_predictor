#!/usr/bin/env bash
# The two best PC predictors and their frozen BP references on another dataset (plan_datasets_precision.md, rev 2).
#   bash job_datasets.sh <project_root> <job_dir> <mnist|cifar10> <gates|prep|train|eval|all> [smoke]
# Everything for dataset X lives in Q1-PhaseA/ds/X (ds/X_smoke for smoke runs), with the same layout as Q1-PhaseA so the
# Galaxy scripts run unchanged from that directory.  Galaxy10 settings everywhere; nothing is re-tuned.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); DS="$3"; STAGE="$4"; SMOKE="${5:-}"
cd "$P/Q1-PhaseA" || exit 2
Q=$(pwd); PY=$(readlink -f ../.venv/bin/python)
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=$Q/datasets_status.txt
note() { echo "$(date +%m-%d_%H:%M) [$DS/$STAGE${SMOKE:+/smoke}] $*" | tee -a "$ST"; }
case "$DS" in mnist|cifar10) ;; *) echo "bad dataset $DS"; exit 2 ;; esac
PAR=${PAR:-3}
note "start at $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- data (idempotent)
mkdir -p ../data
if [ ! -f ../data/MNIST/raw/train-images-idx3-ubyte.gz ]; then          # both datasets: the gates test both
  mkdir -p ../data/MNIST/raw && cp "$JOBDIR"/*-idx?-ubyte.gz ../data/MNIST/raw/ || { note "MNIST copy failed"; exit 3; }
fi
if [ ! -f ../data/cifar10/cifar-10-batches-py/test_batch ]; then
  mkdir -p ../data/cifar10 && ( cd ../data/cifar10 && curl -sSL -o c10.tgz https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz \
    && echo "c58f30108f718f92721af3b95e74349a  c10.tgz" | md5sum -c - && tar xzf c10.tgz && rm c10.tgz ) || { note "CIFAR-10 download failed"; exit 3; }
fi
mkdir -p ds; [ -e ds/data ] || ln -s ../../data ds/data

# ---------------------------------------------------------------- gates (Q1-PhaseA, Galaxy regression included)
if [ "$STAGE" = gates ] || [ "$STAGE" = all ]; then
  $PY -W ignore test_datasets.py > "$JOBDIR/gates_${DS}.log" 2>&1; rc=$?
  note "gates rc=$rc $(grep -c PASS "$JOBDIR/gates_${DS}.log") pass $(grep -c FAIL "$JOBDIR/gates_${DS}.log") fail"; [ $rc -eq 0 ] || exit 4
fi

W=ds/$DS; [ -n "$SMOKE" ] && W=ds/${DS}_smoke
mkdir -p "$W/pred_runs" && cd "$W" || exit 2
collect() { ( cd "$Q" && tar czf "$JOBDIR/ds_${DS}${SMOKE:+_smoke}.tar.gz" --exclude='*.eqx' --exclude='pred_latents.npz' "$W" 2>/dev/null ) || true; }
trap collect EXIT
if [ -n "$SMOKE" ]; then ENC=200; NTR=2048; GST=30; FST=30; else ENC=17200; NTR=-1; GST=4000; FST=6000; fi

# ---------------------------------------------------------------- prep: frozen backbone (feature extractor only) + latents
if [ "$STAGE" = prep ] || [ "$STAGE" = all ]; then
  if [ ! -f runs/bp_sigreg_enc_s0/params.eqx ]; then
    $PY -W ignore $Q/run_arms.py --dataset $DS --arms bp --seeds 0 --tag enc --steps $ENC --out runs > prep_backbone.log 2>&1 \
      || { note "backbone training failed"; exit 5; }
  fi
  $PY -W ignore $Q/ds_gate.py --run runs/bp_sigreg_enc_s0 --out backbone_gate.json > prep_gate.log 2>&1; rc=$?
  note "backbone gate rc=$rc $(tail -1 prep_gate.log | cut -c1-200)"
  [ $rc -eq 0 ] || [ -n "$SMOKE" ] || exit 6                    # smoke backbone is undertrained: report only
  if [ ! -f pred_latents.npz ]; then
    $PY -W ignore $Q/pred_data.py --run runs/bp_sigreg_enc_s0 --out pred_latents.npz --n-train $NTR > prep_latents.log 2>&1 \
      || { note "latent extraction failed"; exit 7; }
  fi
  cp $Q/pred_runs/gpc_selected.json pred_runs/gpc_selected.json   # Galaxy step scales, unchanged (user decision)
  note "prep done: $(du -h pred_latents.npz | cut -f1) latents"
fi

# ---------------------------------------------------------------- train: G, F and the two frozen BP references, 3 seeds
if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
  SEEDS="0 1 2"; [ -n "$SMOKE" ] && SEEDS="0"; EV=""; [ -n "$SMOKE" ] && EV="--smoke"
  mkdir -p logs pred_runs/tests_sep_fix pred_runs/tests pred_runs/ffres_fix pred_runs/frozen_bp_fix pred_runs/tests_ffres_fix/R16 pred_runs/tests_ffres_bpfix
  { for s in $SEEDS; do
      echo "G_s$s|[ -f pred_runs/tests_sep_fix/gpc_sep_d10_amort_hp16_a1_k1_amort_c1_T2200_bp_s$s.json ] || $PY -W ignore $Q/run_gpc_tests.py --rule sep --backbone bp --seed $s --depth 10 --belief amort --start amort --rho 16 --k-in 1 --T2 200 --c 1 --alpha 1 --ckpt --steps $GST $EV --outdir pred_runs/tests_sep_fix"
      echo "BPg_s$s|[ -f pred_runs/tests/gpc_bp_d10_zero_bp_s$s.json ] || $PY -W ignore $Q/run_gpc_tests.py --rule bp --backbone bp --seed $s --depth 10 --start zero --steps $GST $EV --outdir pred_runs/tests"
      echo "F_s$s|[ -f pred_runs/ffres_fix/ffres_almR16_d10_s$s.json ] || $PY -W ignore $Q/pred_ffres.py --arm alm --depth 10 --seed $s --T 400 --alpha 1.0 --lr 0.01 --rho 16 --steps $FST --save --out pred_runs/ffres_fix/ffres_almR16_d10_s$s.json; $PY -W ignore $Q/run_pred_tests.py ffres --arm alm --depth 10 --seed $s --model pred_runs/ffres_fix/ffres_almR16_d10_s$s.eqx --outdir pred_runs/tests_ffres_fix/R16"
      echo "BPf_s$s|[ -f pred_runs/frozen_bp_fix/ffres_bp_d10_s$s.json ] || $PY -W ignore $Q/pred_ffres.py --arm bp --depth 10 --seed $s --lr 0.001 --steps $FST --save --out pred_runs/frozen_bp_fix/ffres_bp_d10_s$s.json; $PY -W ignore $Q/run_pred_tests.py ffres --arm bp --depth 10 --seed $s --model pred_runs/frozen_bp_fix/ffres_bp_d10_s$s.eqx --outdir pred_runs/tests_ffres_bpfix"
    done; } > logs/runs.txt
  run_one() { n="${1%%|*}"; cmd="${1#*|}"; bash -c "set -o pipefail; $cmd" > "logs/$n.log" 2>&1; echo "$(date +%m-%d_%H:%M) rc=$? $n" >> logs/runs_status.txt; }
  export -f run_one
  tr '\n' '\0' < logs/runs.txt | xargs -0 -P $PAR -I{} bash -c 'run_one "$@"' _ {}
  bad=$(grep -vc ' rc=0 ' logs/runs_status.txt || true)
  note "training done: $(grep -c ' rc=0 ' logs/runs_status.txt) ok, $bad failed"
  [ -n "$SMOKE" ] && [ "$bad" != 0 ] && exit 8
fi

# ---------------------------------------------------------------- eval: 1-192 whole views, mean + precision pools, kNN
if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
  if [ -n "$SMOKE" ]; then EA="--n 200 --vmax 12 --seeds 0"; else EA=""; fi
  $PY -W ignore $Q/diag_saccade2.py --whole-only --depths 10 --no-diag --backbone runs/bp_sigreg_enc_s0 \
      --out pred_runs/eval_views.json $EA > logs/eval.log 2>&1; rc=$?
  note "eval rc=$rc"; [ $rc -eq 0 ] || exit 9
fi
note "DONE"
