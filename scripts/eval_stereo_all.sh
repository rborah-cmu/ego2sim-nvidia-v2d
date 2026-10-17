#!/usr/bin/env bash
# Stereo (undistorted cam_a + SGBM depth) for every evaluation episode, sequential, CPU.
source ~/v2d/env.sh
FPY=~/v2d/envs/v2d_foundation_pose/.venv/bin/python
for EP in 1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38; do
  S=~/v2d/ego2sim/ep$EP/stereo; [ -f $S/cam_a_undist.mp4 ] && continue; mkdir -p $S
  $FPY ~/v2d/tools/ego_stereo_depth.py ~/v2d/v2d_data/track_3/public/meta/camera_calibration.json ~/v2d/tests_recon/stereo/extr_ep12.json \
    ~/v2d/v2d_data/track_3/evaluation/videos/chunk-000 $EP $S 2>&1 | grep -v -i warn | tail -1 | sed "s/^/ep$EP /"
done
