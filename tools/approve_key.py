"""Pre-approve the stored API key for the AgentSTAR launcher's dedicated Claude Code state, so the
detached agent isn't stopped by Claude Code's "Detected a custom API key ... use this key?" prompt.
Claude Code records approvals in .claude.json -> customApiKeyResponses.approved as the key's last 20
characters. Run with sudo (the state dir is root-owned):  sudo python3 approve_key.py [CLAUDE_HOME_DIR]
(default: the laptop WSL location). Never prints the key."""
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~rborah/.local/state/claude-articulated/claude-home")
key = open(os.path.join(d, "anthropic_api_key")).read().strip()
suffix = key[-20:]
p = os.path.join(d, ".claude.json")
cfg = json.load(open(p))
resp = cfg.setdefault("customApiKeyResponses", {})
approved = resp.setdefault("approved", [])
if suffix not in approved:
    approved.append(suffix)
resp["rejected"] = [s for s in resp.get("rejected", []) if s != suffix]
tmp = p + ".tmp"
with open(tmp, "w") as f:
    json.dump(cfg, f, indent=2)
os.chmod(tmp, 0o600)
os.replace(tmp, p)
print(f"approved: stored key is now pre-approved ({len(approved)} approved, {len(resp['rejected'])} rejected)")
