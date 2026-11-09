#!/usr/bin/env bash
# Runs on the A4000 host (always on).  Every 15 min: fetch main of the public backup repo and copy every FINISHED run
# (report.json or diverged.json) from each <name>/stageA/ into runs_vit/stageA/ unless already there, so the A4000's own
# sequence skips it.  Never overwrites.  Training-checkpoint branches are not fetched.
set +e; set -u
R=${R:-/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA/runs_vit}; B=${B:-/media/md_lab/hd_21/preprocessed/pc_predictor_runs_backup}
URL=${URL:-https://github.com/YashNagraj75/pc_predictor-runs.git}; EVERY=${EVERY:-900}; DAYS=${DAYS:-6}; JOBDIR=$(pwd)
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/merge_log.txt"; }
end=$(( $(date +%s) + ${LIMIT:-$((DAYS * 86400))} )); log "start"
while [ $(date +%s) -lt $end ]; do
  if [ -d $B/.git ]; then git -C $B fetch -q origin main && git -C $B reset -q --hard origin/main; else git clone -q --single-branch -b main $URL $B; fi
  [ $? -eq 0 ] || log "fetch failed"
  for d in $B/*/stageA/*/; do
    [ -d "$d" ] || continue; n=$(basename "$d"); case $n in gpu_gate*|a0) continue ;; esac
    [ -f "$d/report.json" ] || [ -f "$d/diverged.json" ] || continue
    [ -e "$R/stageA/$n" ] && continue
    cp -a "$d" "$R/stageA/.$n.tmp" && mv "$R/stageA/.$n.tmp" "$R/stageA/$n" && log "merged $n from $(basename $(dirname $(dirname $d)))"
    s=$(dirname "$d")/$n.log; [ -f "$s" ] && [ ! -e "$R/stageA/$n.log" ] && cp "$s" "$R/stageA/$n.log"
  done
  sleep $EVERY
done
log "time limit reached"
