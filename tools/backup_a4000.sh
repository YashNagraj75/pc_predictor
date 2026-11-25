#!/usr/bin/env bash
# Backs up every untracked A4000 result (runs, checkpoints, logs) to pc_predictor-runs/a4000 with the a4000 deploy key, then
# re-clones over HTTPS and checks every file's md5 against the source.  Skips ckpt*.eqx of finished runs, latent states,
# caches and locks; >95 MB files are split.  Work dir on hd_21 (/home is 89% full); GitHub ssh on port 443 (22 is blocked).
set -u
P=${PCP_ROOT:-/media/md_lab/hd_21/preprocessed/pc_predictor}; W=$P/_backup_work; C=$W/repo; V=$W/verify; L=$W/backup_log.txt
REPO=YashNagraj75/pc_predictor-runs; mkdir -p $W
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $L; }
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/pcp_runs_a4000 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -p 443"
G() { git -C $C -c user.name=pc_predictor-a4000 -c user.email=noreply@users.noreply.github.com "$@"; }
retry() { local n=0; until "$@"; do n=$((n + 1)); [ $n -ge 6 ] && return 1; log "retry $n: $*"; sleep 60; done; }
log "start"
if [ ! -d $C/.git ]; then
  retry git clone -q --depth 1 --filter=blob:none --sparse ssh://git@ssh.github.com:443/$REPO.git $C || { log "clone failed"; exit 2; }
fi
G sparse-checkout set a4000 || { log "sparse-checkout failed"; exit 2; }
cd $P || exit 2
{ git ls-files --others --exclude-standard -- Q1-PhaseA logs '_sync_backup_*'          # untracked, not ignored
  git ls-files --others --ignored --exclude-standard -- Q1-PhaseA logs '_sync_backup_*'; } 2>/dev/null | sort -u \
  | grep -vE '(^|/)ckpt[^/]*\.eqx$|^Q1-PhaseA/runs_vit/latent/states/|/__pycache__/|\.lock$|/\.fail_[^/]*$|^\.venv/|^data/' \
  | while read -r f; do [ -f "$f" ] && echo "$f"; done > $W/files.txt
for d in Q1-PhaseA/runs_vit/stageA/*/; do                  # unfinished runs keep their checkpoints
  [ -f "$d/report.json" ] || [ -f "$d/diverged.json" ] || find "$d" -name 'ckpt*.eqx' >> $W/files.txt; done
KB=$(tr '\n' '\0' < $W/files.txt | du -ck --files0-from=- | tail -1 | cut -f1); FREE=$(df -k --output=avail $W | tail -1)
log "files: $(wc -l < $W/files.txt), $((KB / 1024)) MB; free on the work disk $((FREE / 1024)) MB"
[ $FREE -gt $((KB * 4)) ] || { log "not enough free space (need 4x: staging, git objects, verify clone)"; exit 3; }
tr '\n' '\0' < $W/files.txt | xargs -0 md5sum > $W/MANIFEST.md5 || { log "md5 of sources failed"; exit 3; }
mkdir -p $C/a4000 && cp $W/MANIFEST.md5 $C/a4000/MANIFEST.md5
retry rsync -a --partial --files-from=$W/files.txt $P/ $C/a4000/ || { log "staging copy failed"; exit 3; }
find $C/a4000 -type f -size +95M ! -name '*.part_*' | while read -r f; do rm -f "$f".part_*
  split -b 95m -d "$f" "$f.part_" && rm "$f" && log "split $(basename $f)"; done
for g in $(cut -d/ -f1-2 $W/files.txt | sed 's#^_sync_backup_.*#_sync_backups#' | sort -u) a4000/MANIFEST.md5; do
  case $g in _sync_backups) G add a4000/_sync_backup_* ;; a4000/*) G add "$g" ;; *) G add "a4000/$g" ;; esac
  G diff --cached --quiet && continue
  G commit -q -m "a4000 backup: $g" && retry G push -q origin HEAD:main || { log "push failed at $g"; exit 4; }
  log "pushed $g"
done
log "verifying from a fresh HTTPS clone"
rm -rf $V && retry git clone -q --depth 1 --filter=blob:none --sparse https://github.com/$REPO.git $V && git -C $V sparse-checkout set a4000 \
  || { log "verify clone failed"; exit 5; }
cd $V/a4000 && for p in $(find . -name '*.part_00'); do cat "${p%.part_00}".part_* > "${p%.part_00}"; done
md5sum -c --quiet $W/MANIFEST.md5 > $W/verify_failures.txt 2>&1; bad=$(grep -c . $W/verify_failures.txt)
n=$(wc -l < $W/MANIFEST.md5)
if [ $bad -eq 0 ]; then log "BACKUP COMPLETE: all $n files on GitHub match the A4000 (md5)"; else log "BACKUP INCOMPLETE: $bad of $n files missing or different (see verify_failures.txt)"; fi
log "DONE"
