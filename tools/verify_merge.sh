#!/usr/bin/env bash
# Runs on the A4000 next to fetch_merge.sh.  Every 30 min: its own fresh fetch of the backup repo's main, then for every
# FINISHED rented-GPU run checks that the A4000 copy in runs_vit/stageA exists and every file has the same git hash.
# Also reads each rented GPU's BACKUP_VERIFY verdict.  Report: runs_vit/stageA/MERGE_VERIFY.txt.
set +e; set -u
R=${R:-${PCP_ROOT:?set PCP_ROOT, see .env.example}/Q1-PhaseA/runs_vit}; V=${V:-${PCP_VERIFY_DIR:?set PCP_VERIFY_DIR, see .env.example}}
URL=${URL:-https://github.com/YashNagraj75/pc_predictor-runs.git}; EVERY=${EVERY:-1800}; JOBDIR=$(pwd); OUT=$R/stageA/MERGE_VERIFY.txt
end=$(( $(date +%s) + ${LIMIT:-$((6 * 86400))} ))
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/verify_merge_log.txt"; }
check() {
  [ -d $V/.git ] || { git init -q $V && git -C $V remote add origin $URL; }
  git -C $V fetch -q --depth 1 origin +main:refs/remotes/origin/main || { echo "GitHub fetch FAILED"; return 2; }
  git -C $V checkout -q -f refs/remotes/origin/main 2>/dev/null || { echo "checkout FAILED"; return 2; }
  local d n src f rel bad=0 ok=0 wait=0 lines=""
  echo "checked $(date '+%Y-%m-%d %H:%M') (A4000 local time) against main @ $(git -C $V rev-parse --short HEAD)"
  for d in $V/*/stageA/*/; do
    n=$(basename $d); src=$(basename $(dirname $(dirname $d))); case $n in gpu_gate*|a0) continue ;; esac
    [ -f $d/report.json ] || [ -f $d/diverged.json ] || continue
    if [ ! -d $R/stageA/$n ]; then lines+="  waiting $n (from $src): not merged yet (merge runs every 15 min)\n"; wait=$((wait + 1)); continue; fi
    local diff=0 cnt=0
    while IFS= read -r f; do
      rel=${f#$d}; cnt=$((cnt + 1))
      [ -f "$R/stageA/$n/$rel" ] && [ "$(git hash-object "$R/stageA/$n/$rel")" = "$(git hash-object "$f")" ] || diff=$((diff + 1))
    done < <(find $d -type f)
    if [ $diff -eq 0 ]; then lines+="  ok $n (from $src, $cnt files identical)\n"; ok=$((ok + 1))
    else lines+="  DIFFERS $n (from $src): $diff of $cnt files missing or different on the A4000\n"; bad=$((bad + 1)); fi
  done
  echo "finished rented-GPU runs: $ok on the A4000 and identical, $wait not merged yet, $bad different"; printf "$lines"
  for f in $V/*/stageA/BACKUP_VERIFY_*.txt; do [ -f $f ] && echo "rented GPU verdict ($(basename $f .txt | cut -c15-)): $(head -2 $f | tr '\n' ' ')"; done
  [ $bad -eq 0 ]
}
log "start"
while [ $(date +%s) -lt $end ]; do
  check > $OUT.tmp; mv $OUT.tmp $OUT; log "$(sed -n 2p $OUT)"
  sleep $EVERY
done
