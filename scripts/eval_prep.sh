#!/usr/bin/env bash
# Per evaluation episode (after eval_stereo_all.sh): WiLoR over all frames (GPU) -> auto hand prompts -> grid sheet.
#   eval_prep.sh GPU
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=${1:-1}
HOST=~/v2d/envs/recon_host/.venv/bin/python; CV=~/v2d/envs/v2d_sam2/.venv/bin/python
cd ~/v2d/vtd_pinned/reconstruction
for EP in 1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38; do
  E=~/v2d/ego2sim/ep$EP; S=$E/stereo
  while [ ! -f $S/stereo_stats.json ]; do sleep 20; done
  [ -d $E/hands/wilor_raw ] && [ -n "$(ls $E/hands/wilor_raw)" ] || { mkdir -p $E/hands; $HOST -c "
from v2d.wilor.docker.run_video_to_hands import run_video_to_hands
run_video_to_hands(video_path='$S/cam_a_undist.mp4', output_dir='$E/hands/wilor_raw', weights_dir='data/weights/wilor')" > $E/wilor_raw.log 2>&1; }
  [ -f $E/hand_prompts.json ] || $CV ~/v2d/tools/auto_hand_prompts.py $E/hands/wilor_raw $E/hand_prompts.json
  [ -f ~/v2d/ego2sim/ep${EP}_grid.jpg ] || $CV ~/v2d/tools/make_grid.py $S/cam_a_undist.mp4 ~/v2d/ego2sim/ep${EP}_grid.jpg
done