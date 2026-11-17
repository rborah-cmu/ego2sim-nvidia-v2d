"""Sum Claude Code token usage for the AgentSTAR launcher's dedicated state (root-owned, so run with sudo):
    sudo python3 /mnt/d/v2d/agentstar/token_usage.py [SINCE_ISO [UNTIL_ISO]]     e.g. 2026-10-14T23:07 2026-10-15T00:09
Reads ~rborah/.local/state/claude-articulated/claude-home/projects/**/*.jsonl (main agent + sub-agents) and totals
message.usage per model. On riddle: sudo CLAUDE_HOME_DIR=~rborah/v2d/.state/claude-articulated/claude-home python3 token_usage.py ... Prints token counts only (no prompts, no keys). For dollars, use the Anthropic Console."""
import glob
import json
import os
import sys
from collections import defaultdict

since = sys.argv[1] if len(sys.argv) > 1 else ""
until = sys.argv[2] if len(sys.argv) > 2 else ""
root = os.path.join(os.environ.get("CLAUDE_HOME_DIR") or os.path.expanduser("~rborah/.local/state/claude-articulated/claude-home"), "projects")
fields = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
tot = defaultdict(lambda: defaultdict(int))
seen = set()
files = glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
for path in files:
    with open(path, errors="ignore") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            ts = str(rec.get("timestamp", ""))
            if (since and ts < since) or (until and ts > until):
                continue
            msg = rec.get("message") or {}
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            key = (msg.get("id"), rec.get("requestId"))  # streamed messages can repeat; count each once
            if key in seen:
                continue
            seen.add(key)
            model = msg.get("model", "?")
            tot[model]["requests"] += 1
            for k in fields:
                tot[model][k] += int(usage.get(k) or 0)
# $ per million tokens: (input, output, cache read). First-party list prices as of 2026-11; cache writes are billed at
# 1.25x input (5-minute cache). A cache-read price of None is not published here and is taken as 0.1x input.
PRICES = {"claude-fable-5-2": (8.0, 40.0, 0.20), "claude-fable-5-1": (10.0, 50.0, None), "claude-opus-5-5": (4.0, 20.0, 0.20),
          "claude-opus-5": (5.0, 25.0, None), "claude-sonnet-5-5": (2.0, 10.0, 0.10), "claude-sonnet-5": (2.0, 10.0, None),
          "claude-haiku-5-5": (0.10, 0.50, None)}


def dollars(model, t):
    key = next((k for k in sorted(PRICES, key=len, reverse=True) if model.startswith(k)), None)
    if key is None:
        return None
    p_in, p_out, p_read = PRICES[key]
    p_read = 0.1 * p_in if p_read is None else p_read
    return (t["input_tokens"] * p_in + t["cache_creation_input_tokens"] * 1.25 * p_in
            + t["cache_read_input_tokens"] * p_read + t["output_tokens"] * p_out) / 1e6


print(f"{len(files)} transcript files, since={since or 'all time'} until={until or 'now'}")
total = 0.0
for model, t in tot.items():
    d = dollars(model, t)
    total += d or 0.0
    print(f"{model}: {t['requests']} requests | input {t['input_tokens']:,} | cache write {t['cache_creation_input_tokens']:,} | "
          f"cache read {t['cache_read_input_tokens']:,} | output {t['output_tokens']:,} | "
          + (f"~${d:.2f}" if d is not None else "price unknown"))
print(f"estimated total ~${total:.2f} at list prices (models without a listed price are not counted)")
