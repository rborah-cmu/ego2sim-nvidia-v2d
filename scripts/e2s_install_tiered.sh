#!/usr/bin/env bash
# Install the planner + builder (tiered) Ego2Sim harness files into ~/v2d/agenticSTAR (branch ego2sim-shape) from this
# repo checkout. No sudo. Idempotent. The single-model brief and its behaviour are left as they are.
#   bash scripts/e2s_install_tiered.sh [REPO_DIR=this checkout]
set -euo pipefail
SRC=${1:-$(cd "$(dirname "$0")/.." && pwd)}/agentstar_harness
A=~/v2d/agenticSTAR
cd "$A"
git rev-parse --abbrev-ref HEAD | grep -qx ego2sim-shape || { echo "not on branch ego2sim-shape" >&2; exit 1; }
mkdir -p ego2sim/agents ego2sim/notes
cp "$SRC/AGENT_TASK_EGO2SIM_V2.md" AGENT_TASK_EGO2SIM_V2.md
cp "$SRC/agents/shape-planner.md" ego2sim/agents/shape-planner.md
cp "$SRC/contract_check.py" harness/analysis/contract_check.py   # adds the plateau stop (exit 4) and the surface line
python3 -m py_compile harness/analysis/contract_check.py
echo "installed: AGENT_TASK_EGO2SIM_V2.md, ego2sim/agents/shape-planner.md, harness/analysis/contract_check.py"
echo "launch:   E2S_TIERED=1 bash ~/v2d/scripts/launch_e2s_riddle.sh CAPTURE TAG [N_FRAMES] [GPU]"
git status --short | head
