#!/usr/bin/env bash
# Export each finished Ego2Sim AgentSTAR run's mesh to ~/v2d/ego2sim/meshes/<obj>.{obj,glb} (newest run per object).
# wooden_piece_1's mesh also serves wooden_piece_2 (near-identical scans; identity comes from the tracks). Polls until all
# objects exist.   e2s_mesh_watch.sh
P=~/v2d/envs/v2d_foundation_pose/.venv/bin/python; M=~/v2d/ego2sim/meshes; R=~/v2d/agenticSTAR/runs
ALL="white_pot white_pot_lid wooden_spoon beige_cup blue_cup plastic_dish_rack mini_sweeper mini_dust_pan water_pitcher wooden_piece_1 wooden_piece_2"
mkdir -p $M
while true; do
  for d in $(ls -dt $R/ep*_* 2>/dev/null); do
    tag=$(basename $d); obj=$(echo $tag | sed -E 's/^ep[0-9]+_//; s/_[0-9]{8}-[0-9]{6}$//')
    [ -f $M/$obj.obj ] && continue
    st=$(python3 -c "import json;print(json.load(open('$d/supervisor/state.json')).get('status',''))" 2>/dev/null)
    [ "$st" = exited ] || continue
    [ -f $d/mesh/object.glb ] || { echo "$(date +%T) $tag exited without mesh/object.glb"; continue; }
    (cd /tmp && $P ~/v2d/tools/agentstar_mesh_export.py $d/mesh $M/$obj.obj --glb 2>&1 | grep -v -i warn)
    echo "$(date +%T) $tag -> $obj ($(python3 -c "import json;print(json.load(open('$d/contract_status.json')).get('all_pass'))" 2>/dev/null) contract)"
    [ $obj = wooden_piece_1 ] && cp $M/wooden_piece_1.obj $M/wooden_piece_2.obj && cp $M/wooden_piece_1.glb $M/wooden_piece_2.glb
  done
  n=0; for o in $ALL; do [ -f $M/$o.obj ] && n=$((n + 1)); done
  [ $n -eq 11 ] && { echo "$(date +%T) all 11 meshes"; exit 0; }
  sleep 120
done