#!/usr/bin/env bash
# One A4000 job (user 2026-11-12; rev 2 after 3 reviewers): (1) latent analysis of every finished stage-A run (its result
# does NOT gate training), (2) standard-PC baseline spc_u1: smoke, then depth 10 and depth 6 (8 power iterations) via
# run_stageA_all.sh, then a check that both 25-epoch runs exist with hv_iters 8, (3) latent analysis again (adds spc_u1).
# Results are packed on exit, whatever happens.  Log: sequence.txt.  Needs SHA=<short commit>.
set +e; set +o pipefail; set -u
JOBDIR=$(pwd); R=${R:-${PCP_ROOT:?set PCP_ROOT, see .env.example}}; SHA=${SHA:?set SHA}; OUT=$R/Q1-PhaseA/runs_vit/latent; B=$R/Q1-PhaseA/runs_vit/stageA
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$JOBDIR/sequence.txt"; }
pack() {
  tar czf "$JOBDIR/latent_results.tar.gz" -C "$OUT" --exclude=states . ; log "pack latent rc=$?"
  ( cd "$B" && ls -d spc_u1_* >/dev/null 2>&1 && tar czf "$JOBDIR/spc_results.tar.gz" $(ls -d spc_u1_*/ | sed 's#/$##' | while read d; do for f in report.json train_log.json per_image.npz; do [ -f "$d/$f" ] && echo "$d/$f"; done; done) spc_u1_*.log ../../stageA_status.txt ); log "pack spc rc=$?"
}
trap pack EXIT
cd "$R" || exit 2
[ "$(git rev-parse --short HEAD)" = "$SHA" ] || { log "HEAD is $(git rev-parse --short HEAD), expected $SHA"; exit 9; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { log "tracked files modified on the host: $(git status --porcelain --untracked-files=no | head -3)"; exit 9; }
ls -d "$B"/spc_u1_* >/dev/null 2>&1 && { log "STOPPED: spc_u1 run dirs already exist (would be reused): $(ls -d "$B"/spc_u1_* | head -3)"; exit 9; }
log "sequence start at $SHA"
bash Q1-PhaseA/job_latent_vit.sh "$R" "$JOBDIR" "$OUT" 1; rc1=$?; log "latent analysis (1) rc=$rc1 (does not gate training)"
cd "$JOBDIR" && SHA=$SHA STAGES="smoke_spc depth:10 depth:6" ARMS_OVERRIDE="spc_u1" R=$R bash run_stageA_all.sh; rc=$?; cd "$R"
log "spc_u1 training rc=$rc"
ok=1
for D in 10 6; do
  d=$(ls -d "$B"/spc_u1_d${D}_lr*_e25_full_s0_* 2>/dev/null | head -1)
  if [ -n "$d" ] && [ -f "$d/report.json" ] && [ -f "$d/predictor.eqx" ] && "$R/.venv/bin/python" -c "import json,sys; sys.exit(0 if json.load(open('$d/report.json'))['cfg'].get('hv_iters') == 8 else 1)"; then
    log "spc_u1 d$D final run present ($(basename $d), hv_iters 8)"
  else ok=0; log "spc_u1 d$D final run MISSING or not hv_iters 8 ($d)"; fi
done
[ $ok -eq 1 ] || { log "STOPPED before latent analysis (2): spc_u1 training incomplete (training rc=$rc)"; exit 7; }
bash Q1-PhaseA/job_latent_vit.sh "$R" "$JOBDIR" "$OUT" 2; rc2=$?; log "latent analysis (2) rc=$rc2"
if [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]; then log "ALL DONE"; exit 0; else log "FINISHED WITH ERRORS (latent 1 rc=$rc1, latent 2 rc=$rc2)"; exit 1; fi
