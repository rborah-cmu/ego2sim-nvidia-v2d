#!/usr/bin/env bash
# exit 0 when the last AgentSTAR mesh (wooden_spoon) is exported and GPU 0 is completely free
[ -f ~/v2d/ego2sim/meshes/wooden_spoon.obj ] || exit 1
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | grep -c .)" -eq 0 ] || exit 1
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -le 500 ]