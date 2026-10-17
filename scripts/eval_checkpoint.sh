#!/usr/bin/env bash
# Run the official FlashCHORD evaluator (pinned commit) on one checkpoint (riddle).
# Usage: eval_checkpoint.sh GPU CHECKPOINT OUTSTEM [WORLDS]
# Writes OUTSTEM.json (metrics) and OUTSTEM.parquet (object trajectories), like the kit's runner.
set -euo pipefail
GPU=$1; CKPT=$(realpath "$2"); STEM=$3; WORLDS=${4:-1}
source ~/v2d/env.sh
cd ~/v2d/vtd_pinned/robotic_grounding/flash_chord
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$(dirname "$STEM")"
SECONDS=0
python ~/v2d/submission_kit/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py \
  "evaluation.checkpoint=$CKPT" "evaluation.world_count=$WORLDS" evaluation.reset_mode=explicit \
  "evaluation.metrics_output=$STEM.json" "evaluation.object_trajectories_output=$STEM.parquet" \
  "hydra.run.dir=$STEM.hydra" logging.mode=disabled
echo "eval wall ${SECONDS}s"
