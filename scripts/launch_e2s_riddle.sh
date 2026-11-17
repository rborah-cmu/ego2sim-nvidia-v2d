#!/usr/bin/env bash
# Launch one Ego2Sim shape-only AgentSTAR run on riddle (Opus 5.5 unless ARTSCRIPT_CLAUDE_MODEL is set).
#   bash ~/v2d/scripts/launch_e2s_riddle.sh mini_dust_pan_e2s ep40_mini_dust_pan [N_FRAMES=8] [GPU=0]
# Refuses to start unless the GPU is completely free (lab rule). Detached in root's tmux; 1.5 h cap.
set -euo pipefail
source ~/v2d/env.sh
CAPTURE=${1:?capture name, e.g. mini_dust_pan_e2s}
TAG=${2:?spec tag, e.g. ep40_mini_dust_pan}
N=${3:-8}
GPU=${4:-0}
export XDG_CONFIG_HOME=~/v2d/.config
cd ~/v2d/agenticSTAR
git rev-parse --abbrev-ref HEAD | grep -qx ego2sim-shape || { echo "not on branch ego2sim-shape" >&2; exit 1; }
[ -f "ego2sim/specs/$TAG.json" ] || { echo "missing spec ego2sim/specs/$TAG.json" >&2; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | grep -c . || true)
if [ "$used" -gt 500 ] || [ "$procs" -gt 0 ]; then
  echo "GPU $GPU is in use (${used} MiB, ${procs} processes) - pick another GPU or wait." >&2; exit 1
fi
FRAMES=$(python3 - "captures/$CAPTURE/capture.json" "$N" <<'PY'
import json, sys
fr = json.load(open(sys.argv[1]))["frames"]; n = min(int(sys.argv[2]), len(fr))
print(",".join(fr[round(i * (len(fr) - 1) / max(n - 1, 1))] for i in range(n)))
PY
)
# per-run brief: ego2sim/briefs/$TAG.md if present (generic brief + object notes), else the generic brief
TEMPLATE="$PWD/AGENT_TASK_EGO2SIM.md"; [ -f "ego2sim/briefs/$TAG.md" ] && TEMPLATE="$PWD/ego2sim/briefs/$TAG.md"
# E2S_TIERED=1: planner + builder. The builder (the main loop, where most tokens go) runs on a cheaper model and
# calls the shape-planner sub-agent on a stronger one for the plan and when stuck (AGENT_TASK_EGO2SIM_V2.md).
#   E2S_WORKER_MODEL (default claude-sonnet-5-5), E2S_PLANNER_MODEL (default claude-opus-5-5);
#   object notes, if any, are appended from ego2sim/notes/$TAG.md.
if [ "${E2S_TIERED:-0}" = 1 ]; then
  [ -f AGENT_TASK_EGO2SIM_V2.md ] && [ -f ego2sim/agents/shape-planner.md ] || { echo "tiered files missing (run e2s_install_tiered.sh)" >&2; exit 1; }
  export ARTSCRIPT_CLAUDE_MODEL=${E2S_WORKER_MODEL:-claude-sonnet-5-5}
  mkdir -p .claude/agents ego2sim/briefs
  sed "s/^model: .*/model: ${E2S_PLANNER_MODEL:-claude-opus-5-5}/" ego2sim/agents/shape-planner.md > .claude/agents/shape-planner.md
  TEMPLATE="$PWD/ego2sim/briefs/${TAG}_tiered.md"
  cat AGENT_TASK_EGO2SIM_V2.md > "$TEMPLATE"
  [ -f "ego2sim/notes/$TAG.md" ] && { printf '\n## Object notes\n' >> "$TEMPLATE"; cat "ego2sim/notes/$TAG.md" >> "$TEMPLATE"; }
  echo "tiered: builder $ARTSCRIPT_CLAUDE_MODEL, planner ${E2S_PLANNER_MODEL:-claude-opus-5-5}"
fi
echo "capture $CAPTURE, spec $TAG, GPU $GPU, frames $FRAMES, brief $TEMPLATE"
echo "start (UTC): $(date -u +%Y-%m-%dT%H:%M)"
# The agent starts in root's tmux server. If that server is already running, a new session takes the server's
# environment, not this shell's, so the model has to be set (or cleared) there too.
if [ -n "${ARTSCRIPT_CLAUDE_MODEL:-}" ]; then
  sudo tmux set-environment -g ARTSCRIPT_CLAUDE_MODEL "$ARTSCRIPT_CLAUDE_MODEL" 2>/dev/null || true
else
  sudo tmux set-environment -g -u ARTSCRIPT_CLAUDE_MODEL 2>/dev/null || true
fi
sudo env HOME="$HOME" PATH="$PATH" MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" XDG_CONFIG_HOME="$XDG_CONFIG_HOME" \
  BLENDER="$HOME/v2d/blender/blender" SANDBOX_EGRESS_HOSTS="api.anthropic.com:443 platform.claude.com:443" \
  ${ARTSCRIPT_CLAUDE_MODEL:+ARTSCRIPT_CLAUDE_MODEL="$ARTSCRIPT_CLAUDE_MODEL"} \
  ./run.sh --capture "captures/$CAPTURE" --frames "$FRAMES" "$TAG" \
    --task-template "$TEMPLATE" --contract "ego2sim/specs/$TAG.json" \
    --side-by-side-max-dimension 1024 --candidate-sheet-max-dimension 1024 \
    --gpu "$GPU" --workers 4 --agent claude --timeout-hours 1.5 --detach
