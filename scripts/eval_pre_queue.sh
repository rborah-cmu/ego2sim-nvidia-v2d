#!/usr/bin/env bash
# Mesh-independent stages (SAM2, MoGe, fused depth, VO, hands) for evaluation episodes, in sequence on one GPU.
# Objects come from the kit's roster (object id i = i-th roster object). Writes obj_prompts.json from obj_prompts.py first.
#   eval_pre_queue.sh GPU EP [EP ...]
set -uo pipefail
GPU=$1; shift
source ~/v2d/env.sh
ROSTER=~/v2d/submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json
python3 ~/v2d/tools/obj_prompts.py ~/v2d/ego2sim > /dev/null
for EP in "$@"; do
  OBJS=$(python3 -c "import json;print(','.join(json.load(open('$ROSTER'))['episodes']['$EP']))")
  until [ -f ~/v2d/ego2sim/ep$EP/hand_prompts.json ]; do sleep 30; done
  echo "$(date +%T) ep$EP ($OBJS)"
  STOP_BEFORE_FP=1 SPLIT=evaluation DEVICE=cuda:0 bash ~/v2d/scripts/ego2sim_public_episode.sh "$EP" "$OBJS" "$GPU" 2>&1 | grep -v -i warn
done
echo "$(date +%T) done"