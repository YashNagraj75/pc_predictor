#!/usr/bin/env bash
# Runs on the A4000 host (always on): every 15 min, pull a rented GPU's stage-A results over a read-only rrsync key into
# runs_vit/stageA_<name>/, then copy every FINISHED run (report.json or diverged.json) that the A4000 does not have yet
# into runs_vit/stageA/, so the A4000's own sequence skips it.  Never overwrites.  Stops after the remote sequence ends.
# bash backup_pull.sh <name> <host> <port>
set +e; set -u
NAME=$1; H=$2; PORT=$3; R=/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA/runs_vit
DST=$R/stageA_$NAME; mkdir -p $DST $R/${NAME}_logs; JOBDIR=$(pwd)
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$JOBDIR/backup_log.txt" >> $DST/_backup_log.txt; }
SSH="ssh -i $HOME/.ssh/pcp_gpu_backup -p $PORT -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_pcp"
sync_once() {
  rsync -a --partial -e "$SSH" ubuntu@$H:stageA/ $DST/ 2>>"$JOBDIR/rsync_err.txt"; local a=$?
  rsync -a -e "$SSH" ubuntu@$H:${NAME}_logs/ $R/${NAME}_logs/ 2>>"$JOBDIR/rsync_err.txt"; local b=$?
  [ $a -eq 0 ] && [ $b -eq 0 ]
}
merge() {
  for d in $DST/*/; do
    n=$(basename "$d"); case $n in gpu_gate*|a0) continue ;; esac
    [ -f "$d/report.json" ] || [ -f "$d/diverged.json" ] || continue
    [ -e "$R/stageA/$n" ] && continue
    cp -a "$d" "$R/stageA/.$n.tmp" && mv "$R/stageA/.$n.tmp" "$R/stageA/$n" && log "merged $n"
    [ -f "$DST/$n.log" ] && [ ! -e "$R/stageA/$n.log" ] && cp "$DST/$n.log" "$R/stageA/$n.log"
  done
}
log "start: $NAME = $H:$PORT"; fails=0
while :; do
  if sync_once; then fails=0; merge; else fails=$((fails + 1)); log "sync failed ($fails in a row)"; fi
  if grep -qE "ALL STAGES DONE|STOPPED" $R/${NAME}_logs/sequence.txt 2>/dev/null; then
    sleep 60; sync_once && merge; log "remote sequence ended: $(tail -1 $R/${NAME}_logs/sequence.txt)"; exit 0
  fi
  sleep 900
done
