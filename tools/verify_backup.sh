#!/usr/bin/env bash
# Runs on a rented GPU next to backup_push.sh.  Every 15 min it checks GitHub itself (its own HTTPS fetch, not the push
# loop's copy): every pushable file must be on main with an identical git hash, and each unfinished run's ckpt.eqx must be
# on ckpt-<name>.  Report: runs_vit/stageA/BACKUP_VERIFY_<name>.txt (pushed with the results).  After the stage job ends
# it re-pushes until everything matches, then writes "SAFE TO DELETE: YES".   bash verify_backup.sh <name> <sequence.txt>
set +e; set -u
NAME=$1; SEQ=$2; Q=${Q:-$HOME/pc_predictor/Q1-PhaseA}; URL=${URL:-https://github.com/YashNagraj75/pc_predictor-runs.git}
V=${V:-$HOME/runs_verify}; EVERY=${EVERY:-900}; JOBDIR=$(pwd); PUSH=${PUSH:-$JOBDIR/backup_push.sh}
S=$Q/runs_vit/stageA; OUT=$S/BACKUP_VERIFY_$NAME.txt
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/verify_log.txt"; }
fetch() {
  [ -d $V/.git ] || { git init -q $V && git -C $V remote add origin $URL; }
  git -C $V fetch -q --depth 1 origin +main:refs/remotes/origin/main || return 1
  git -C $V fetch -q --depth 1 origin +ckpt-$NAME:refs/remotes/origin/ckpt 2>/dev/null || git -C $V update-ref -d refs/remotes/origin/ckpt 2>/dev/null
  return 0
}
step_of() { python3 -c "import json,sys; l=json.load(sys.stdin); print(l[-1].get('step','?') if l else '?')" 2>/dev/null || echo "?"; }
verify() {   # report on stdout; rc 0 = everything on this machine is on GitHub
  fetch || { echo "GitHub fetch FAILED"; return 2; }
  declare -A M K; local mode type hash path n d f rel h bad=0 nf=0 nu=0 lines=""
  while IFS=$' \t' read -r mode type hash path; do M[$path]=$hash; done < <(git -C $V ls-tree -r refs/remotes/origin/main -- $NAME/stageA)
  while IFS=$' \t' read -r mode type hash path; do K[$path]=$hash; done < <(git -C $V ls-tree -r refs/remotes/origin/ckpt 2>/dev/null)
  echo "checked $(date -u '+%Y-%m-%d %H:%M UTC') against $URL main @ $(git -C $V rev-parse --short refs/remotes/origin/main)"
  for d in $S/*/; do
    n=$(basename $d); case $n in gpu_gate_*|.*) continue ;; esac
    if [ -f $d/ckpt.eqx ] && [ ! -f $d/report.json ] && [ ! -f $d/diverged.json ]; then      # unfinished run
      nu=$((nu + 1)); h=$(git hash-object $d/ckpt.eqx); ls=$(step_of < $d/train_log.json)
      if [ "${K[$n/ckpt.eqx]:-}" = "$h" ]; then lines+="  UNFINISHED $n: latest checkpoint (step $ls) is on GitHub\n"
      elif [ -n "${K[$n/ckpt.eqx]:-}" ]; then
        gs=$(git -C $V show refs/remotes/origin/ckpt:$n/train_log.json 2>/dev/null | step_of)
        lines+="  UNFINISHED $n: GitHub checkpoint is older (GitHub step $gs, local $ls); next push within 15 min\n"; bad=$((bad + 1))
      else lines+="  UNFINISHED $n: NO checkpoint on GitHub yet\n"; bad=$((bad + 1)); fi
      continue
    fi
    local miss=0 diff=0 cnt=0
    while IFS= read -r f; do
      rel=${f#$S/}; h=$(git hash-object "$f"); cnt=$((cnt + 1))
      if [ -z "${M[$NAME/stageA/$rel]:-}" ]; then miss=$((miss + 1)); elif [ "${M[$NAME/stageA/$rel]}" != "$h" ]; then diff=$((diff + 1)); fi
    done < <(find $d -type f ! -name 'ckpt*.eqx' ! -name '.gpu.lock' ! -path '*/.fail_*'; [ -f $S/$n.log ] && echo $S/$n.log)
    [ -f $d/report.json ] || [ -f $d/diverged.json ] && nf=$((nf + 1))
    if [ $miss -eq 0 ] && [ $diff -eq 0 ]; then lines+="  ok $n ($cnt files)\n"
    else lines+="  NOT YET ON GITHUB $n: $miss missing, $diff different of $cnt files\n"; bad=$((bad + 1)); fi
  done
  echo "finished runs: $nf, unfinished runs: $nu, problems: $bad"; printf "$lines"
  [ $bad -eq 0 ]
}
log "start: $NAME, watching $SEQ"
while :; do
  if grep -qE "ALL STAGES DONE|STOPPED" "$SEQ" 2>/dev/null; then
    log "stage job ended: $(tail -1 $SEQ)"; ok=1
    for i in 1 2 3 4 5 6 7 8; do
      verify > $OUT.tmp; ok=$?; mv $OUT.tmp $OUT; [ $ok -eq 0 ] && break
      log "end check $i: $(sed -n 2p $OUT); re-pushing"
      B=$HOME/runs_backup_v C=$HOME/runs_ckpt_v EVERY=5 bash $PUSH $NAME $SEQ; sleep 60
    done
    { [ $ok -eq 0 ] && echo "SAFE TO DELETE: YES (every result and checkpoint on this machine is on GitHub)" \
                    || echo "SAFE TO DELETE: NO (see the problems below; do not delete this instance)"; cat $OUT; } > $OUT.tmp && mv $OUT.tmp $OUT
    B=$HOME/runs_backup_v C=$HOME/runs_ckpt_v EVERY=5 bash $PUSH $NAME $SEQ                 # push the verdict itself
    fetch && [ "$(git -C $V ls-tree refs/remotes/origin/main -- $NAME/stageA/BACKUP_VERIFY_$NAME.txt | awk '{print $3}')" = "$(git hash-object $OUT)" ] \
      && log "verdict on GitHub: $(head -1 $OUT)" || log "verdict NOT confirmed on GitHub: $(head -1 $OUT)"
    exit $ok
  fi
  verify > $OUT.tmp; rc=$?; mv $OUT.tmp $OUT; log "check: $(sed -n 2p $OUT)"
  sleep $EVERY
done
