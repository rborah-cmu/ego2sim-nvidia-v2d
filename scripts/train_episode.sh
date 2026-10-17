#!/usr/bin/env bash
# FlashSAC (sharpa_flash_sac recipe) on one retargeted ManoSharpaData partition.
#   train_episode.sh GPU PARTITION_DIR NAME [WORLDS=2048] [STEPS=8388608] [REPLAY_CAP] [REPLAY_MIN]
# PARTITION_DIR = .../sequence_id=<id>/robot_name=sharpa_wave. STEPS must be divisible by WORLDS.
# Output ~/v2d/runs/NAME (train.log, gpu.csv, checkpoints). Same knobs as smoke_train_tissue.sh.
# MOTION_SPEED (default 1.0; the recipe uses 0.5): with the parquet at 20 fps this is one evaluator step per video frame.
# TRAIN_LOG=offline: local W&B run under OUT/wandb (no network, no uploads) for reward/termination curves
#   (read with tools/read_wandb_offline.py); default disabled.
# EXTRA: extra Hydra overrides (space-separated), e.g. EXTRA="sim.njmax=1024 sim.nconmax=512" for multi-object scenes.
set -euo pipefail
GPU=${1:?gpu}; P=$(realpath "${2:?partition dir}"); NAME=${3:?name}
WORLDS=${4:-2048}; STEPS=${5:-8388608}
CAP=${6:-$((WORLDS * 1024))}; MINSZ=${7:-$((WORLDS * 100))}
SAVE=${SAVE_EVERY:-$(( (STEPS / 2 / WORLDS) * WORLDS ))}  # SAVE_EVERY: checkpoint interval (multiple of WORLDS)
source ~/v2d/env.sh
cd ~/v2d/vtd_pinned/robotic_grounding/flash_chord
. .venv/bin/activate
OUT=~/v2d/runs/$NAME; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 5 -i "$GPU" > "$OUT/gpu.csv" &
SMI=$!
LOGARGS="logging.mode=disabled"
if [ "${TRAIN_LOG:-disabled}" = offline ]; then
  export WANDB_DIR="$OUT" WANDB_SILENT=true WANDB_MODE=offline
  LOGARGS="logging.mode=offline logging.upload_checkpoints=false logging.upload_training_state=false logging.run_name=$NAME"
fi
SECONDS=0
set +e
python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${P}'" reset.seed=42 $LOGARGS \
  task.motion_speed="${MOTION_SPEED:-1.0}" scene.world_count="$WORLDS" training.total_environment_steps="$STEPS" \
  training.replay.capacity="$CAP" training.replay.minimum_size="$MINSZ" \
  training.checkpoint.save_interval_environment_steps="$SAVE" \
  output_dir="$OUT" ${EXTRA:-} 2>&1 | tee "$OUT/train.log"
RC=${PIPESTATUS[0]}
kill $SMI
echo "exit $RC wall ${SECONDS}s" | tee -a "$OUT/train.log"
