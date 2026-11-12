#!/usr/bin/env bash
# One host job for ALL remaining stage-A work, in order: depth 10, depth 6, BP noise floor (STAGES overrides, e.g.
# STAGES="gpugate:10 depth:6" on a second GPU).  Each stage is re-run after a
# clean time-limit stop (rc 75) and resumes from its checkpoints; rc 1 (stage finished, some arms recorded as failed)
# moves on; any other rc (GPU gate, lock, sync, crash) stops for a decision.  The host runs job commands with -e, so
# this script turns it off and never depends on a line running after a non-zero command.  Log: sequence.txt.
set +e; set +o pipefail; set -u
JOBDIR=$(pwd); R=${R:-/media/md_lab/hd_21/preprocessed/pc_predictor}; SHA=${SHA:-8ee37d9}; PASS_MIN=${PASS_MIN:-1380}
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/sequence.txt"; }
cd "$R" || { log "cannot cd to $R"; exit 2; }
[ "$(git rev-parse --short HEAD)" = "$SHA" ] || { log "HEAD is $(git rev-parse --short HEAD), expected $SHA"; exit 9; }
P=$(pgrep -af 'pytho[n].*(train_pred_vit|a0_vit|vit_ijepa|gpu_gate)[.]py' | head -3)
[ -z "$P" ] || { log "stale training processes, not starting: $P"; exit 8; }

run_stage() {  # label, then the job_stageA.sh arguments
  local label=$1; shift; local n=0 rc
  while [ $n -lt 40 ]; do
    n=$((n + 1)); log "$label: pass $n start"
    JOB_MIN=$PASS_MIN bash Q1-PhaseA/job_stageA.sh "$R" "$JOBDIR" "$@"; rc=$?
    log "$label: pass $n rc=$rc"
    case $rc in
      0) return 0 ;;
      75) sleep 30 ;;
      1) log "$label: finished with recorded arm failures (see stageA_status.txt); continuing"; return 0 ;;
      *) return $rc ;;
    esac
  done
  log "$label: 40 passes without finishing"; return 70
}

log "sequence start at $(git rev-parse --short HEAD), stages: ${STAGES:-depth:10 depth:6 noise}"
for st in ${STAGES:-depth:10 depth:6 noise}; do
  mode=${st%%:*}; depth=${st#*:}; [ "$depth" = "$st" ] && depth=""
  case $mode in
    smoke_spc) bash Q1-PhaseA/job_stageA.sh "$R" "$JOBDIR" smoke_spc; rc=$?; log "smoke_spc rc=$rc"; [ $rc -eq 0 ] || { log "STOPPED: smoke_spc failed"; exit 5; } ;;
    gpugate) log "gpu gate d$depth"; bash Q1-PhaseA/job_stageA.sh "$R" "$JOBDIR" gpugate $depth; rc=$?
             log "gpu gate d$depth rc=$rc"; [ $rc -eq 0 ] || { log "STOPPED: GPU gate failed"; exit 4; } ;;
    depth)   if [ -n "${ARMS_OVERRIDE:-}" ]; then export STAGEA_ARMS="$ARMS_OVERRIDE"
             elif [ "$depth" = 10 ]; then export STAGEA_ARMS="${STAGEA_ARMS:-bp epc_k4 epc_k16 epc_alm_blk4 epc_alm_k4}"; else unset STAGEA_ARMS; fi
             run_stage "depth $depth" depth $depth; rc=$?; unset STAGEA_ARMS; [ $rc -eq 0 ] || { log "STOPPED in depth $depth, rc=$rc"; exit $rc; } ;;
    noise)   run_stage "noise" noise; rc=$?; [ $rc -eq 0 ] || { log "STOPPED in noise, rc=$rc"; exit $rc; } ;;
    *)       log "unknown stage $st"; exit 64 ;;
  esac
done
log "ALL STAGES DONE"
