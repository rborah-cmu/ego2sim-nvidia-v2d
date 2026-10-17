#!/usr/bin/env bash
# Evaluation-episode worker for one GPU: claims the next episode whose meshes all exist (mkdir lock), runs the full
# Ego2Sim driver with our meshes, trains the locked recipe (1x), and retrains at 2x when MPPE vs our reference > 3 cm.
# Results: ~/v2d/logs/e2s_results_eval.tsv (one row per run; pick the lowest ref MPPE per episode when packing).
#   eval_worker.sh GPU WAIT_CMD EP [EP ...]     (WAIT_CMD: shell test that must pass before starting, e.g. "true")
set -uo pipefail
GPU=$1; WAITC=$2; shift 2
source ~/v2d/env.sh
M=~/v2d/ego2sim/meshes; L=~/v2d/ego2sim/claims; RES=~/v2d/logs/e2s_results_eval.tsv; mkdir -p $L
ROSTER=~/v2d/submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json
until eval "$WAITC"; do sleep 60; done
echo "$(date +%T) worker GPU $GPU starts"
objs() { python3 -c "import json;print(','.join(json.load(open('$ROSTER'))['episodes']['$1']))"; }
ready() { local o; for o in ${1//,/ }; do [ -f $M/$o.obj ] || return 1; done; }
left=("$@")
while [ ${#left[@]} -gt 0 ]; do
  pick=""; rest=()
  for e in "${left[@]}"; do
    if [ -z "$pick" ] && ready "$(objs $e)" && mkdir $L/ep$e 2>/dev/null; then pick=$e; else
      [ -d $L/ep$e ] || rest+=("$e"); fi
  done
  left=("${rest[@]}")
  [ -z "$pick" ] && { [ ${#left[@]} -eq 0 ] && break; sleep 120; continue; }
  O=$(objs $pick); echo "$(date +%T) ep$pick ($O) on GPU $GPU"
  SPLIT=evaluation MESH_SRC=$M DEVICE=cuda:0 bash ~/v2d/scripts/ego2sim_public_episode.sh $pick $O $GPU 2>&1 | grep -v -i warn
  [ -d ~/v2d/ego2sim/train/ep$pick/processed ] || { echo "ep$pick: no training partition, skipped"; continue; }
  SCALE=1 SUFFIX=_eval RES=$RES bash ~/v2d/scripts/e2s_train_queue_v2.sh $GPU $pick:$O
  m=$(awk -v e=$pick '$1==e {print $4}' $RES | tail -1)
  if awk -v m="${m:-99}" 'BEGIN{exit !(m > 3.0)}'; then
    echo "$(date +%T) ep$pick ref MPPE $m cm > 3 -> 2x retry"
    SCALE=2 SUFFIX=_eval2 RES=$RES bash ~/v2d/scripts/e2s_train_queue_v2.sh $GPU $pick:$O
  fi
done
echo "$(date +%T) worker GPU $GPU done"