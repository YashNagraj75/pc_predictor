#!/usr/bin/env bash
# Local end-to-end test of backup_push.sh + fetch_merge.sh against a bare repo, under the host wrapper's bash -e.
set -u
T=$(mktemp -d); TOOLS=$(cd "$(dirname "$0")" && pwd)
git init -q --bare -b main $T/remote.git
git clone -q $T/remote.git $T/seed && (cd $T/seed && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m init && git push -q origin HEAD:main)
Q=$T/l40/Q1-PhaseA; S=$Q/runs_vit/stageA; mkdir -p $S/a_d6 $S/b_d6 $S/c_d6 $S/gpu_gate_1 $T/job $T/a4000/runs_vit/stageA/x_d10
echo r > $S/a_d6/report.json; echo w > $S/a_d6/predictor.eqx; echo ck > $S/a_d6/ckpt.eqx; echo log > $S/a_d6.log
echo ck > $S/b_d6/ckpt.eqx; echo ckp > $S/b_d6/ckpt_prev.eqx; echo tl > $S/b_d6/train_log.json
echo dv > $S/c_d6/diverged.json; echo g > $S/gpu_gate_1/x.npy; echo st > $Q/stageA_status.txt
echo "running" > $T/job/sequence.txt; mkdir -p $T/l40/x_d10; echo orig > $T/a4000/runs_vit/stageA/x_d10/report.json
cd $T/job
echo "bash backup_push.sh l40a $T/job/sequence.txt" > cmd.sh; cp $TOOLS/backup_push.sh .
( sleep 4; echo "11-09 ALL STAGES DONE" >> $T/job/sequence.txt ) &
env REPO=$T/remote.git Q=$Q B=$T/bk C=$T/ck EVERY=2 GIT_SSH_COMMAND=ssh timeout 60 bash -eo pipefail ./cmd.sh; echo "push exit $?"
cat backup_log.txt | cut -c13-
echo "--- main branch files:"; git --git-dir=$T/remote.git ls-tree -r --name-only main | sort
echo "--- ckpt-l40a branch files:"; git --git-dir=$T/remote.git ls-tree -r --name-only ckpt-l40a | sort
echo "--- ckpt branch commits: $(git --git-dir=$T/remote.git rev-list --count ckpt-l40a)"
mkdir -p $T/mjob && cd $T/mjob && cp $TOOLS/fetch_merge.sh . && echo "bash fetch_merge.sh" > cmd.sh
env R=$T/a4000/runs_vit B=$T/a4000/bk URL=$T/remote.git EVERY=1 LIMIT=3 timeout 30 bash -eo pipefail ./cmd.sh; echo "merge exit $?"
cut -c13- merge_log.txt; echo "--- A4000 stageA now:"; ls $T/a4000/runs_vit/stageA | tr '\n' ' '; echo
echo "a_d6 has ckpt? $(ls $T/a4000/runs_vit/stageA/a_d6 | tr '\n' ' ')"
rm -rf $T
