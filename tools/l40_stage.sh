#!/usr/bin/env bash
# Rented-GPU stage-A job.  On a fresh instance: clone the code at 8ee37d9, install the A4000's exact package versions,
# fetch CIFAR-10 and the stage-A inputs from branch data-cifar10 of the public backup repo (seconds), check every md5
# against the A4000 copy, then run the given STAGES with run_stageA_all.sh.  Logs are mirrored into runs_vit/<name>_logs.
set +e; set -u
JOBDIR=$(pwd); S=$HOME/pc_predictor; Q=$S/Q1-PhaseA; NAME=${NAME:-l40}
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/sequence.txt"; }
MD5S="$JOBDIR/cifar10_a4000.md5"; [ -f "$MD5S" ] || { log "missing cifar10_a4000.md5 input"; exit 2; }
[ -d $S/.git ] || git clone -q https://github.com/YashNagraj75/pc_predictor.git $S || { log "code clone failed"; exit 2; }
cd $S && git checkout -q 8ee37d9 2>/dev/null || { git fetch -q origin && git checkout -q 8ee37d9; }
[ "$(git rev-parse --short HEAD)" = 8ee37d9 ] || { log "repo not at 8ee37d9"; exit 9; }
if [ ! -x $S/.venv/bin/python ]; then
  uv venv -q --python 3.13 $S/.venv && UV_LINK_MODE=copy uv pip install -q --python $S/.venv/bin/python "jax==0.11.0" "jaxlib==0.11.0" \
    "jax-cuda12-plugin[with-cuda]==0.11.0" "jax-cuda12-pjrt==0.11.0" "equinox==0.13.8" "optax==0.2.8" "numpy==2.5.3" "jaxtyping==0.3.11" \
    "scipy==1.18.1" "h5py==3.16.0" "ml-dtypes==0.6.0" > "$JOBDIR/pip.log" 2>&1 || { log "package install failed"; exit 3; }
fi
DB=$HOME/data_cifar10; C=$S/data/cifar10; IN="$JOBDIR/l40_inputs.tar.gz"
if [ ! -f $C/cifar-10-batches-py/data_batch_1 ] || [ ! -f "$IN" ]; then
  [ -d $DB/.git ] || git clone -q --depth 1 -b data-cifar10 https://github.com/YashNagraj75/pc_predictor-runs.git $DB || { log "data branch clone failed"; exit 2; }
  [ -f $C/cifar-10-batches-py/data_batch_1 ] || { mkdir -p $C && cp -a $DB/cifar10/cifar-10-batches-py $C/; }
  [ -f "$IN" ] || cp $DB/inputs/l40_inputs.tar.gz "$IN"
  [ "$(md5sum "$IN" | cut -c1-32)" = "$(cut -c1-32 $DB/inputs/MD5SUMS)" ] || { log "inputs checksum mismatch"; exit 2; }
fi
( cd $C/cifar-10-batches-py && md5sum -c --quiet "$MD5S" ) || { log "CIFAR-10 differs from the A4000 copy"; exit 2; }
tar xzf "$IN" -C $Q || { log "inputs unpack failed"; exit 2; }
log "data and inputs ready"
mkdir -p $Q/runs_vit/${NAME}_logs
( while :; do cp "$JOBDIR/sequence.txt" $Q/stageA_status.txt $Q/runs_vit/${NAME}_logs/ 2>/dev/null; sleep 120; done ) &
MIRROR=$!
cp "$JOBDIR/run_stageA_all.sh" . && R=$S bash run_stageA_all.sh; rc=$?
cp "$JOBDIR/sequence.txt" $Q/stageA_status.txt $Q/runs_vit/${NAME}_logs/ 2>/dev/null; kill $MIRROR 2>/dev/null
exit $rc
