#!/usr/bin/env bash
# Run Ego2Sim AgentSTAR launches one after another on ONE GPU, reserved for them. Run it in your own terminal (the
# launcher calls sudo; it may ask for the password again when sudo's timestamp expires). For each CAPTURE:TAG it waits
# until the capture + spec exist and the GPU is completely free (the launcher's own rule, unchanged), then launches.
#   bash ~/v2d/scripts/e2s_launch_queue.sh GPU CAPTURE:TAG [CAPTURE:TAG ...]
set -uo pipefail
GPU=$1; shift
A=~/v2d/agenticSTAR
free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -le 500 ] && \
         [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | grep -c .)" -eq 0 ]; }
for x in "$@"; do
  cap=${x%%:*}; tag=${x#*:}
  until [ -f "$A/captures/$cap/capture.json" ] && [ -f "$A/ego2sim/specs/$tag.json" ]; do sleep 60; done
  until free; do sleep 60; done
  echo "$(date +%T) launching $cap ($tag) on GPU $GPU"
  until bash ~/v2d/scripts/launch_e2s_riddle.sh "$cap" "$tag" 8 "$GPU"; do sleep 120; until free; do sleep 60; done; done
  sleep 300   # let the run take the GPU before checking for free again
done
echo "$(date +%T) all launched"