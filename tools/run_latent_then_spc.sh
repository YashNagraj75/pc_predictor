#!/usr/bin/env bash
# One A4000 job, in order (user 2026-11-12): (1) latent analysis of every finished stage-A run, (2) standard-PC baseline
# spc_u1 at depth 10 then depth 6 (8 power iterations for the step size), via run_stageA_all.sh, (3) latent analysis again,
# which adds the spc_u1 runs.  Log: sequence.txt (+ latent_sequence.txt).  Needs SHA=<short commit>.
set +e; set +o pipefail; set -u
JOBDIR=$(pwd); R=${R:-/media/md_lab/hd_21/preprocessed/pc_predictor}; SHA=${SHA:?set SHA}; OUT=$R/Q1-PhaseA/runs_vit/latent
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/sequence.txt"; }
cd "$R" || exit 2
[ "$(git rev-parse --short HEAD)" = "$SHA" ] || { log "HEAD is $(git rev-parse --short HEAD), expected $SHA"; exit 9; }
log "latent analysis (1) start"; bash Q1-PhaseA/job_latent_vit.sh "$R" "$JOBDIR" "$OUT"; rc=$?; log "latent analysis (1) rc=$rc"
[ $rc -eq 0 ] || { log "STOPPED after latent analysis (1)"; exit $rc; }
tar czf "$JOBDIR/latent_results_1.tar.gz" -C "$OUT" --exclude=states . 2>/dev/null; log "latent results packed"
cd "$JOBDIR" && SHA=$SHA STAGES="depth:10 depth:6" ARMS_OVERRIDE="spc_u1" R=$R bash run_stageA_all.sh; rc=$?; cd "$R"
log "spc_u1 training rc=$rc"; [ $rc -eq 0 ] || { log "STOPPED in spc_u1 training"; exit $rc; }
log "latent analysis (2) start"; bash Q1-PhaseA/job_latent_vit.sh "$R" "$JOBDIR" "$OUT"; rc=$?; log "latent analysis (2) rc=$rc"
tar czf "$JOBDIR/latent_results_2.tar.gz" -C "$OUT" --exclude=states . 2>/dev/null
log "ALL DONE"; exit $rc
