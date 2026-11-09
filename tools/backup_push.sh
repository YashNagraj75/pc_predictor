#!/usr/bin/env bash
# Runs on a rented GPU next to its stage-A job.  Every 15 min: commit run RESULTS (reports, per-image eval, final weights,
# logs; no training checkpoints) to <repo>/<name>/ on main, and force-push the latest checkpoints of unfinished runs to
# branch ckpt-<name> (a single commit, no history).  The deploy key can write only to the backup repo.  Stops after the
# stage job's sequence log ends.   bash backup_push.sh <name> <path to the stage job's sequence.txt>
set +e; set -u
NAME=$1; SEQ=$2; REPO=${REPO:-git@github.com:YashNagraj75/pc_predictor-runs.git}; Q=${Q:-$HOME/pc_predictor/Q1-PhaseA}
B=${B:-$HOME/runs_backup}; C=${C:-$HOME/runs_ckpt}; EVERY=${EVERY:-900}; JOBDIR=$(pwd)
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -i $HOME/.ssh/pcp_runs_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new}"
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/backup_log.txt"; }
G() { git -c user.name="$NAME-backup" -c user.email=noreply@users.noreply.github.com "$@"; }
[ -d $B/.git ] || git clone -q $REPO $B || { log "clone failed"; exit 2; }
sync_results() {
  mkdir -p $B/$NAME/stageA $B/$NAME/logs
  rsync -a --exclude 'ckpt*.eqx' --exclude 'gpu_gate_*' --exclude '.gpu.lock' --exclude '.fail_*' $Q/runs_vit/stageA/ $B/$NAME/stageA/
  cp "$SEQ" $Q/stageA_status.txt $B/$NAME/logs/ 2>/dev/null
  ( cd $B && G add -A $NAME && { G diff --cached --quiet || G commit -q -m "$NAME $(date '+%m-%d %H:%M')"; }
    for i in 1 2 3; do G pull -q --rebase origin main && git push -q origin HEAD:main && exit 0; sleep 20; done; exit 1 )
}
push_ckpts() {
  rm -rf $C; mkdir -p $C; local n=0 d t
  for d in $Q/runs_vit/stageA/*/; do
    [ -f $d/ckpt.eqx ] && [ ! -f $d/report.json ] && [ ! -f $d/diverged.json ] || continue
    t=$C/$(basename $d); mkdir -p $t; cp $d/ckpt.eqx $d/ckpt_prev.eqx $d/train_log.json $t/ 2>/dev/null; n=$((n + 1))
  done
  [ $n -gt 0 ] || return 0
  ( cd $C && git init -q && G add -A && G commit -q -m "latest checkpoints of $n unfinished run(s), $NAME $(date '+%m-%d %H:%M')" \
    && git push -q -f $REPO HEAD:refs/heads/ckpt-$NAME )
}
log "start: $NAME -> $REPO"
while :; do
  sync_results && log "results pushed" || log "results push FAILED"
  push_ckpts || log "checkpoint push FAILED"
  if grep -qE "ALL STAGES DONE|STOPPED" "$SEQ" 2>/dev/null; then sleep 30; sync_results; log "stage job ended, final push: $(tail -1 $SEQ)"; exit 0; fi
  sleep $EVERY
done
