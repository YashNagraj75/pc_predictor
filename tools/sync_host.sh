#!/usr/bin/env bash
# Bring the host checkout to GitHub main without losing host-only content.  bash sync_host.sh <root> <jobdir> <expected_sha>
set -uo pipefail
R="$1"; JOBDIR="$2"; NEW="$3"; cd "$R" || exit 2
REP="$JOBDIR/sync_report.txt"; say() { echo "$*" | tee -a "$REP"; }
git config user.name >/dev/null || git config user.name pc_predictor; git config user.email >/dev/null || git config user.email noreply@users.noreply.github.com
OLD=$(git rev-parse HEAD 2>/dev/null); say "host HEAD before: $OLD"
timeout 600 git fetch -q https://github.com/YashNagraj75/pc_predictor.git main || { say "FETCH FAILED"; exit 3; }
F=$(git rev-parse FETCH_HEAD); say "fetched: $F"; [ "$F" = "$NEW" ] || { say "fetched sha != expected $NEW"; exit 3; }
AHEAD=$(git rev-list --count FETCH_HEAD..HEAD); say "host commits not on GitHub: $AHEAD"
if [ "$AHEAD" != 0 ]; then git log --oneline FETCH_HEAD..HEAD | tee -a "$REP"; git bundle create "$JOBDIR/host_only.bundle" FETCH_HEAD..HEAD; say "ABORT: host has its own commits; bundled for review"; exit 4; fi
# clear the fragile assume-unchanged flags so the reset sees the true state
git ls-files -v | awk '/^h /{print substr($0,3)}' > "$JOBDIR/assume_unchanged.txt"; say "assume-unchanged paths: $(wc -l < "$JOBDIR/assume_unchanged.txt")"
[ -s "$JOBDIR/assume_unchanged.txt" ] && tr '\n' '\0' < "$JOBDIR/assume_unchanged.txt" | xargs -0 git update-index --no-assume-unchanged --
git status --short > "$JOBDIR/status_before.txt"; say "status lines before: $(wc -l < "$JOBDIR/status_before.txt") (modified tracked: $(grep -c '^ M' "$JOBDIR/status_before.txt"))"
B="$R/_sync_backup_$(date +%Y%m%d_%H%M)"; n=0
# (a) files that NEW tracks and that exist here with different content; (b) files tracked now but absent from NEW
{ git ls-tree -r --name-only "$NEW"; git ls-files; } | sort -u | while IFS= read -r f; do
  [ -f "$f" ] || continue
  want=$(git rev-parse -q --verify "$NEW:$f" 2>/dev/null || echo none)
  have=$(git hash-object "$f")
  if [ "$want" != "$have" ]; then mkdir -p "$B/$(dirname "$f")"; cp -p "$f" "$B/$f"; echo "$f" >> "$JOBDIR/backed_up.txt"; fi
done
say "backed up (would change or vanish): $( [ -f "$JOBDIR/backed_up.txt" ] && wc -l < "$JOBDIR/backed_up.txt" || echo 0) files -> $B"
git reset -q --hard "$NEW" || { say "RESET FAILED"; exit 5; }
say "host HEAD after: $(git rev-parse HEAD)"
git status --short > "$JOBDIR/status_after.txt"; say "status after: $(grep -vc '^??' "$JOBDIR/status_after.txt") tracked changes, $(grep -c '^??' "$JOBDIR/status_after.txt") untracked (not ignored)"
grep '^??' "$JOBDIR/status_after.txt" | head -40 >> "$REP" || true   # head closes the pipe early (SIGPIPE under pipefail)
exit 0
