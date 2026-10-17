#!/usr/bin/env bash
# Locked training recipe (NOTES "Phase 5h") for "EP:OBJ1,OBJ2" items, in sequence on one GPU:
# train (40.27M steps, 50M-compressed curriculum, ckpt every 4.19M, offline logs) -> official evaluator on every
# checkpoint -> pick the best by MPPE vs OUR reference -> (public episodes) score vs mocap with the kit's metric code.
# One TSV row per episode -> ~/v2d/logs/e2s_results.tsv
#   e2s_train_queue.sh GPU 2:white_pot_lid,white_pot 40:mini_dust_pan ...
# Env: SUFFIX (run name e2s_epEP_40M$SUFFIX, default empty), RES (results TSV, default ~/v2d/logs/e2s_results.tsv)
set -uo pipefail
GPU=$1; shift
TH="[3958784,6926336,9895936,12863488,15833088,18800640,21770240,24737792,27707392,40265728,50331648]"
RES=${RES:-~/v2d/logs/e2s_results.tsv}
[ -f "$RES" ] || printf "episode\tobjects\tbest_ckpt\tref_mppe_cm\tref_add_auc\tmocap_AUC\tmocap_SP-SR\tmocap_MP-SR\tmocap_RPE\tmocap_MPPE\n" > "$RES"
SC=~/v2d/v2d_data/track_3/public/mesh
for item in "$@"; do
  EP=${item%%:*}; OBJS=${item#*:}; NAME=e2s_ep${EP}_40M${SUFFIX:-}; R=~/v2d/runs/$NAME
  P=~/v2d/ego2sim/train/ep$EP/processed/sequence_id=ep$EP/robot_name=sharpa_wave
  echo "$(date +%T) train ep$EP on GPU $GPU"
  [ -f "$R/policy_40265728.safetensors" ] || TRAIN_LOG=offline SAVE_EVERY=4194304 \
    EXTRA="sim.njmax=1024 sim.nconmax=512 curriculum.thresholds=$TH" \
    bash ~/v2d/scripts/train_episode.sh "$GPU" "$P" "$NAME" 2048 40265728 > /dev/null 2>&1
  echo "$(date +%T) $(tail -1 $R/train.log)"
  first=${OBJS%%,*}
  bash ~/v2d/scripts/eval_all_checkpoints.sh "$R" "$GPU" "$first" > "$R/eval_table.txt" 2>&1
  best=$(awk 'NR>1 && $2+0>0 {print $2, $1}' "$R/eval_table.txt" | sort -n | head -1)
  bmppe=${best%% *}; bstep=${best##* }
  bauc=$(awk -v s="$bstep" '$1==s {print $3}' "$R/eval_table.txt")
  meshes=(); sym=()
  IFS=',' read -ra NS <<< "$OBJS"
  for n in "${NS[@]}"; do
    m=$SC/$n/$n.glb; [ -f "$m" ] || m=$SC/$n/${n}_visual.glb; meshes+=(--mesh "$n=$m")
    case $n in *cup) sym+=(--axial "$n");; white_pot_lid) sym+=(--order2 "$n");; esac
  done
  # roster order for public episodes = sorted names (the dataset's object order)
  roster=$(printf "%s\n" "${NS[@]}" | sort | paste -sd,)
  score=$(cd /tmp && ~/v2d/envs/kit/.venv/bin/python ~/v2d/tools/score_public_episode.py \
    ~/v2d/submission_kit/v2d_submission_kit "$EP" "$R/eval/official_$bstep.parquet" --objects "$roster" \
    "${meshes[@]}" "${sym[@]}" 2>/dev/null | awk '/^  [A-Z]/ {printf "%s\t", $2}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$EP" "$OBJS" "$bstep" "$bmppe" "$bauc" "$score" >> "$RES"
  echo "$(date +%T) ep$EP best $bstep (ref MPPE $bmppe) mocap: $score"
done
