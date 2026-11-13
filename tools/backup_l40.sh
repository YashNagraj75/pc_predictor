#!/usr/bin/env bash
# Pull new/changed stage-A outputs from the rented L40 to this laptop every INTERVAL seconds (incremental rsync over ssh,
# using the `gpu-cloud` entry in ~/.ssh/config; no credentials go to the rented machine).  Keeps resumable checkpoints, so
# an unfinished arm can restart elsewhere if the instance is deleted.  Start:  nohup bash tools/backup_l40.sh &
# Stop:  kill $(cat ~/pc_predictor_l40_backup/.pid).   Log: ~/pc_predictor_l40_backup/backup.log
set -u
HOSTALIAS=${HOSTALIAS:-gpu-cloud}; INTERVAL=${INTERVAL:-600}
DST=${DST:-$HOME/pc_predictor_l40_backup}; mkdir -p "$DST"; echo $$ > "$DST/.pid"
LOG="$DST/backup.log"; log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
command -v rsync > /dev/null || { log "rsync is not installed on this laptop"; exit 1; }
log "backup loop started (every ${INTERVAL}s) -> $DST"
while true; do
  ok=1
  # every run folder (checkpoints, predictors, reports, per-image results) and the per-run logs
  rsync -a --partial --timeout=300 -e "$SSH" "$HOSTALIAS:pc_predictor/Q1-PhaseA/runs_vit/" "$DST/runs_vit/" >> "$LOG" 2>&1 || ok=0
  # stage-A status file and the sequence logs of every job on the instance
  rsync -a --timeout=120 -e "$SSH" "$HOSTALIAS:pc_predictor/Q1-PhaseA/stageA_status.txt" "$DST/" >> "$LOG" 2>&1 || ok=0
  rsync -a --timeout=120 -e "$SSH" --include='*/' --include='sequence.txt' --include='gpu_gate*.log' --include='report_*.json' \
        --exclude='*' "$HOSTALIAS:.claude-science-scratch/.claude-science/jobs/" "$DST/jobs/" >> "$LOG" 2>&1 || ok=0
  if [ $ok -eq 1 ]; then
    log "sync ok: $(find "$DST/runs_vit" -name report.json 2>/dev/null | wc -l) finished runs, $(du -sh "$DST" | cut -f1) total"
    date '+%m-%d %H:%M:%S' > "$DST/LAST_OK"
  else
    log "sync FAILED (instance unreachable or deleted?); last good sync: $(cat "$DST/LAST_OK" 2>/dev/null || echo never)"
  fi
  sleep "$INTERVAL"
done
