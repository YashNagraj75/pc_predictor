#!/usr/bin/env bash
# Second-GPU stage-A job: fetch CIFAR-10 (checksum-verified, identical to the A4000 copy), unpack the V0 encoders and
# A0/per-block decision files from the A4000, then run the given STAGES with run_stageA_all.sh.  Logs are mirrored into
# runs_vit/<name>_logs every 2 min so the A4000's backup loop (backup_pull.sh) copies them with the results.
set +e; set -u
JOBDIR=$(pwd); S=$HOME/pc_predictor; Q=$S/Q1-PhaseA; NAME=${NAME:-l40}
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/sequence.txt"; }
cd $S && [ "$(git rev-parse --short HEAD)" = 8ee37d9 ] || { log "repo not at 8ee37d9"; exit 9; }
C=$S/data/cifar10
if [ ! -f $C/cifar-10-batches-py/data_batch_1 ]; then
  mkdir -p $C && curl -sL -o $C/c.tgz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  [ "$(md5sum $C/c.tgz | cut -c1-32)" = c58f30108f718f92721af3b95e74349a ] || { log "CIFAR-10 checksum mismatch"; exit 2; }
  tar xzf $C/c.tgz -C $C && rm $C/c.tgz
fi
[ "$(md5sum $C/cifar-10-batches-py/data_batch_1 | cut -c1-32)" = c99cafc152244af753f735de768cd75f ] || { log "CIFAR-10 differs from the A4000 copy"; exit 2; }
tar xzf "$JOBDIR/l40_inputs.tar.gz" -C $Q || { log "inputs unpack failed"; exit 2; }
log "data and inputs ready"
mkdir -p $Q/runs_vit/${NAME}_logs
( while :; do cp "$JOBDIR/sequence.txt" $Q/stageA_status.txt $Q/runs_vit/${NAME}_logs/ 2>/dev/null; sleep 120; done ) &
MIRROR=$!
cp "$JOBDIR/run_stageA_all.sh" . && R=$S bash run_stageA_all.sh; rc=$?
cp "$JOBDIR/sequence.txt" $Q/stageA_status.txt $Q/runs_vit/${NAME}_logs/ 2>/dev/null; kill $MIRROR 2>/dev/null
exit $rc
