#!/usr/bin/env bash
# Runs the remaining stage-A stages back to back on the host, resuming after every clean time-limit stop (rc 75),
# so no part waits for an agent to resubmit it.  Waits for a running part ($1 = its job dir) to end first.
set -uo pipefail
JOBDIR=$(pwd); R=/media/md_lab/hd_21/preprocessed/pc_predictor; PREV="$1"
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$JOBDIR/chain.txt"; }
log "waiting for $PREV"
until grep -q "rc=" "$PREV/summary.txt" 2>/dev/null; do sleep 120; done
log "previous part: $(tail -1 $PREV/summary.txt)"
sleep 60
cd $R && [ "$(git rev-parse --short HEAD)" = "8ee37d9" ] || { log HEAD_MISMATCH; exit 9; }
for stage in "depth 10" "depth 6" "noise"; do
  while true; do
    P=$(pgrep -af 'pytho[n].*(train_pred_vit|a0_vit|vit_ijepa|gpu_gate)[.]py' | head -3 || true)
    [ -z "$P" ] || { log "STALE PROCESSES, stopping: $P"; exit 8; }
    log "start $stage"
    JOB_MIN=630 bash Q1-PhaseA/job_stageA.sh $R "$JOBDIR" $stage; rc=$?
    log "$stage rc=$rc"
    [ $rc -eq 75 ] && continue
    [ $rc -eq 0 ] && break
    log "stopping on rc=$rc for a decision"; exit $rc
  done
done
log "ALL STAGES DONE"
