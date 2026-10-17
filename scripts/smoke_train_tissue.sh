#!/usr/bin/env bash
# Short FlashSAC smoke run on the bundled tissue-box floating-Sharpa example (riddle).
# Usage: smoke_train_tissue.sh GPU WORLDS STEPS OUTNAME [REPLAY_CAPACITY] [REPLAY_MIN]
# STEPS and the save interval must be divisible by WORLDS. Recipe defaults: capacity 4194304, min 409600.
set -euo pipefail
GPU=${1:-0}; WORLDS=${2:-512}; STEPS=${3:-2097152}; NAME=${4:-smoke_tissue_w${WORLDS}}
CAP=${5:-524288}; MINSZ=${6:-51200}
SAVE=$(( (STEPS / 2 / WORLDS) * WORLDS ))
source ~/v2d/env.sh
cd ~/v2d/vtd_pinned/robotic_grounding/flash_chord
. .venv/bin/activate
P=$(ls -d ../source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave)
P=$(realpath "$P")
OUT=~/v2d/runs/$NAME; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 5 -i "$GPU" > "$OUT/gpu.csv" &
SMI=$!
SECONDS=0
set +e
python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${P}'" reset.seed=42 logging.mode=disabled \
  scene.world_count="$WORLDS" training.total_environment_steps="$STEPS" \
  training.replay.capacity="$CAP" training.replay.minimum_size="$MINSZ" \
  training.checkpoint.save_interval_environment_steps="$SAVE" \
  output_dir="$OUT" 2>&1 | tee "$OUT/train.log"
RC=${PIPESTATUS[0]}
kill $SMI
echo "exit $RC wall ${SECONDS}s" | tee -a "$OUT/train.log"
