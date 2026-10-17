#!/usr/bin/env bash
# One-time: register the AgentSTAR Claude sandbox launcher on riddle, keeping every file under ~/v2d.
# Run as rborah in your own SSH session (asks for your sudo password and your Anthropic API key):
#   bash ~/v2d/scripts/e2s_install_riddle.sh
# - Runs the upstream installer AS ROOT so bwrap can create namespaces without the AppArmor profile the installer
#   would otherwise write to /etc/apparmor.d (a system security change on a shared machine). It checks this first and
#   stops before installing anything if the check fails.
# - State (settings, transcripts, API key) -> ~/v2d/.state/claude-articulated/claude-home (root-owned, 0700)
#   launcher -> ~/v2d/bin/claude-articulated ; paths.conf -> ~/v2d/.config/claude-articulated/
# - The key is read without echo and written via stdin (never on a command line other users could see in ps).
set -euo pipefail
source ~/v2d/env.sh
A=~/v2d/agenticSTAR
STATE=~/v2d/.state/claude-articulated/claude-home
export XDG_CONFIG_HOME=~/v2d/.config
ENVV=(HOME="$HOME" PATH="$PATH" MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" XDG_CONFIG_HOME="$XDG_CONFIG_HOME")

echo "== 1/4 check (as root) that bwrap works without any AppArmor change"
sudo env "${ENVV[@]}" "$A/tools/install_claude_bwrap.sh" --check --project "$A" --blender ~/v2d/blender \
  --mamba-root "$MAMBA_ROOT_PREFIX" --claude-state "$STATE" --install-dir ~/v2d/bin ||
  { echo "check failed: NOT installing (would need a system AppArmor change). Tell Claude." >&2; exit 1; }

echo "== 2/4 install launcher + state (as root, no system files)"
sudo env "${ENVV[@]}" "$A/tools/install_claude_bwrap.sh" --project "$A" --blender ~/v2d/blender \
  --mamba-root "$MAMBA_ROOT_PREFIX" --claude-state "$STATE" --install-dir ~/v2d/bin --model claude-opus-5-5 2>&1 |
  grep -v "no ANTHROPIC_API_KEY exported" || true

echo "== 3/4 API key (input hidden)"
read -rsp "Paste the Anthropic API key, then Enter: " KEY; echo
[ -n "$KEY" ] || { echo "empty key" >&2; exit 1; }
printf '%s\n' "$KEY" | sudo tee "$STATE/anthropic_api_key" >/dev/null
unset KEY
sudo chmod 600 "$STATE/anthropic_api_key"

echo "== 4/4 pre-approve the key for the detached agent"
sudo python3 ~/v2d/tools/approve_key.py "$STATE"
sudo ls -la "$STATE" | head -8
echo "done. Launch with: bash ~/v2d/scripts/launch_e2s_riddle.sh <capture> <spec tag> [frames] [gpu]"
