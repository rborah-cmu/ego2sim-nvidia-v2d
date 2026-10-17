#!/usr/bin/env bash
# Rebuild Ego2Sim references (ego2sim_public_episode.sh) then train/eval/score them with the locked recipe.
#   e2s_rebuild_train.sh GPU SUFFIX RES EP:OBJS [EP:OBJS ...]
set -uo pipefail
GPU=$1; SUFFIX=$2; RES=$3; shift 3
for x in "$@"; do
  echo "$(date +%T) rebuild ep${x%%:*}"
  DEVICE=cuda:0 bash ~/v2d/scripts/ego2sim_public_episode.sh "${x%%:*}" "${x#*:}" "$GPU" || { echo "rebuild ep${x%%:*} FAILED"; exit 1; }
done
SUFFIX=$SUFFIX RES=$RES bash ~/v2d/scripts/e2s_train_queue.sh "$GPU" "$@"