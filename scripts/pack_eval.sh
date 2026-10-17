#!/usr/bin/env bash
# Assemble the Track 3 submission inputs from the evaluation runs:
#   results/episode_XXXXXX.parquet  = official evaluator parquet of the best checkpoint over runs e2s_epEP_40M_eval{,2}
#                                     (lowest MPPE vs OUR reference, from each run's eval_table.txt)
#   meshes/episode_XXXXXX/<obj>.glb = the simulator's installed object asset (= rollout body coordinates)
# then runs the kit packer.   pack_eval.sh COMMIT_URL [OUT_DIR]
set -uo pipefail
URL=${1:?commit url}; OUT=${2:-$HOME/v2d/submission}
K=~/v2d/submission_kit/v2d_submission_kit; PY=~/v2d/envs/kit/.venv/bin/python
A=~/v2d/vtd_pinned/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed
ROSTER=$K/data/track_3_evaluation_objects.json
rm -rf $OUT/results $OUT/meshes; mkdir -p $OUT/results $OUT/meshes
for EP in $(python3 -c "import json;print(' '.join(json.load(open('$ROSTER'))['episodes']))"); do
  best=""; bm=999
  for R in ~/v2d/runs/e2s_ep${EP}_40M_eval ~/v2d/runs/e2s_ep${EP}_40M_eval2; do
    [ -f $R/eval_table.txt ] || continue
    read s m < <(awk 'NR>1 && $2+0>0 {print $1, $2}' $R/eval_table.txt | sort -k2 -n | head -1)
    [ -n "${m:-}" ] && awk -v a=$m -v b=$bm 'BEGIN{exit !(a<b)}' && { bm=$m; best=$R/eval/official_$s.parquet; }
  done
  if [ -z "$best" ]; then echo "ep$EP: NO RUN"; continue; fi
  cp $best $OUT/results/$(printf "episode_%06d.parquet" $EP)
  D=$OUT/meshes/$(printf "episode_%06d" $EP); mkdir -p $D
  for o in $(python3 -c "import json;print(' '.join(json.load(open('$ROSTER'))['episodes']['$EP']))"); do
    $PY -c "import trimesh; trimesh.load('$A/$o.obj', force='mesh').export('$D/$o.glb')" 2>/dev/null || echo "ep$EP: mesh $o FAILED"
  done
  echo "ep$EP: $(basename $(dirname $(dirname $best)))/$(basename $best) ref MPPE $bm"
done
cd $K && $PY tools/pack_submission.py --track track_3 --results $OUT/results --meshes $OUT/meshes \
  --code-commit "$URL" --eval-script ~/v2d/vtd_pinned/robotic_grounding/flash_chord/scripts/evaluate_policy.py \
  -o $OUT/track3_submission.parquet 2>&1 | tail -25