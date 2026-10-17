#!/usr/bin/env bash
# Evaluate every policy_<steps>.safetensors of a training run with the official evaluator and print one table
# (metrics vs OUR reference, i.e. usable for checkpoint selection on evaluation episodes too).
#   eval_all_checkpoints.sh RUN_DIR GPU [LIFT_BODY]
# LIFT_BODY (optional): also print that body's max achieved vs reference height gain (did it get lifted?).
set -euo pipefail
R=${1:?run dir}; GPU=${2:?gpu}; BODY=${3:-}
for c in $(ls "$R"/policy_*.safetensors | sed 's/.*policy_\([0-9]*\).safetensors/\1/' | sort -n); do
  out="$R/eval/official_$c"
  [ -f "$out.json" ] || bash ~/v2d/scripts/eval_checkpoint.sh "$GPU" "$R/policy_$c.safetensors" "$out" > /dev/null 2>&1
done
~/v2d/envs/kit/.venv/bin/python - "$R" "$BODY" <<'PY'
import glob, json, re, sys
import numpy as np, pyarrow.parquet as pq
R, body = sys.argv[1], sys.argv[2]
print(f"{'steps':>11} {'MPPE cm':>8} {'ADD-AUC':>8} {'RPE cm':>7}  per-body pos err (cm) / rot err (deg)" + ("   lift achieved/ref (cm)" if body else ""))
for js in sorted(glob.glob(f"{R}/eval/official_*.json"), key=lambda p: int(re.findall(r"_(\d+)\.json", p)[0])):
    m = json.load(open(js))["metrics"]
    steps = int(re.findall(r"_(\d+)\.json", js)[0])
    per = " ".join(f"{p*100:5.1f}/{r*57.2958:5.1f}" for p, r in zip(m["object_position_error_per_body_m"], m["object_orientation_error_per_body_rad"]))
    lift = ""
    if body:
        t = pq.read_table(js[:-5] + ".parquet").to_pandas()
        c = t[t.object_body_name == body].sort_values("reference_frame")
        a, r = c.achieved_position_z.to_numpy(), c.reference_position_z.to_numpy()
        lift = f"   {(a.max()-a[0])*100:5.1f}/{(r.max()-r[0])*100:5.1f}"
    print(f"{steps:>11} {m['mppe_cm']:8.2f} {m['add_auc']:8.3f} {m['relative_position_error_cm']:7.2f}  {per}{lift}")
PY
