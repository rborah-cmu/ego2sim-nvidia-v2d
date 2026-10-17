#!/usr/bin/env bash
# FoundationPose on the undistorted ep 12 cam_a video for cup + rack with one depth variant, then compare to mocap.
# Usage: fp_ep12_variant.sh GPU DEPTH_SUBDIR
set -uo pipefail
GPU=$1; DEP=$2
source ~/v2d/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
O=~/v2d/tests_recon/stereo/ep12
M=~/v2d/v2d_data/track_3/public/mesh
H=~/v2d/envs/recon_host/.venv/bin/python
cd ~/v2d/vtd_pinned/reconstruction
SECONDS=0
for spec in "0 blue_cup" "1 plastic_dish_rack"; do
  set -- $spec
  $H -c "
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
run_video_to_poses('$O/cam_a_undist.mp4', '$O/$DEP', '$O/masks/$1', '$O/intrinsics_undist.json', '$M/$2/$2.glb',
                   '$O/poses_${DEP}_$2', 'data/weights/foundation_pose', backend='nvlabs_pytorch',
                   target_width=1014, target_height=760, mask_depth=True, reregister_iou_thresh=0.3)" \
    > "$O/fp_${DEP}_$2.log" 2>&1 || echo "FAILED $2 (see $O/fp_${DEP}_$2.log)"
done
echo "== $DEP (${SECONDS}s)"
$H -I ~/v2d/tools/compare_relpose.py ~/v2d/v2d_data/track_3/public/data/chunk-000/episode_000012.parquet \
  "$O/poses_${DEP}_blue_cup" blue_cup "$O/poses_${DEP}_plastic_dish_rack" plastic_dish_rack --axis-symmetric-a
