#!/usr/bin/env bash
# Score every episode in e2s_results.tsv with achieved := reference (reconstruction only, no RL) vs mocap.
cd /tmp; SC=~/v2d/v2d_data/track_3/public/mesh; PY=~/v2d/envs/kit/.venv/bin/python
tail -n +2 ~/v2d/logs/e2s_results.tsv | while IFS=$'\t' read -r EP OBJS B rest; do
  R=~/v2d/runs/e2s_ep${EP}_40M; $PY ~/v2d/tools/ref_as_rollout.py $R/eval/official_$B.parquet $R/eval/refonly.parquet
  meshes=(); sym=(); IFS=, read -ra NS <<< "$OBJS"
  for n in "${NS[@]}"; do m=$SC/$n/$n.glb; [ -f $m ] || m=$SC/$n/${n}_visual.glb; meshes+=(--mesh "$n=$m"); case $n in *cup) sym+=(--axial $n);; white_pot_lid) sym+=(--order2 $n);; esac; done
  roster=$(printf "%s\n" "${NS[@]}" | sort | paste -sd,)
  s=$($PY ~/v2d/tools/score_public_episode.py ~/v2d/submission_kit/v2d_submission_kit $EP $R/eval/refonly.parquet --objects $roster "${meshes[@]}" "${sym[@]}" 2>/dev/null | awk '/^  [A-Z]/ {printf "%s ", $2}')
  echo "ep$EP refonly AUC SP MP RPE MPPE: $s"
done