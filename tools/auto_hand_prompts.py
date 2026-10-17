"""SAM2 hand prompts from WiLoR detections (no hand-placed boxes): every --every frames (and the last frame), one box
prompt per confident detection, each with a NEW object_id (sides are voted later by ego_hands.py and merged per side).
  auto_hand_prompts.py WILOR_RAW_DIR OUT_JSON [--every 60] [--min-score 0.5] [--min-area 15000] [--pad 0.1]"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("out")
ap.add_argument("--every", type=int, default=60); ap.add_argument("--min-score", type=float, default=0.5)
ap.add_argument("--min-area", type=float, default=15000); ap.add_argument("--pad", type=float, default=0.1)
a = ap.parse_args()
files = sorted(Path(a.raw).glob("*.json"))
n = len(files)
want = sorted(set(list(range(0, n, a.every)) + [n - 1]))
prompts, oid = [], 0
for f0 in want:
    # nearest frame (within +-10) that has detections
    for df in [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 10, -10]:
        f = f0 + df
        if not 0 <= f < n:
            continue
        dets = [d for d in json.load(open(files[f])) if d.get("score", 1) >= a.min_score]
        dets = [d for d in dets if (d["bbox"]["x1"] - d["bbox"]["x0"]) * (d["bbox"]["y1"] - d["bbox"]["y0"]) >= a.min_area]
        if dets:
            break
    for d in dets:
        b = d["bbox"]; W, H = d.get("image_size", [2028, 1520])
        px, py = a.pad * (b["x1"] - b["x0"]), a.pad * (b["y1"] - b["y0"])
        box = {"x0": int(max(0, b["x0"] - px)), "y0": int(max(0, b["y0"] - py)),
               "x1": int(min(W - 1, b["x1"] + px)), "y1": int(min(H - 1, b["y1"] + py))}
        prompts.append({"frame_index": int(files[f].stem), "object_id": oid, "role": "human", "points": [], "point_labels": [], "box": box})
        oid += 1
json.dump({"prompts": prompts}, open(a.out, "w"))
print(f"{a.out}: {oid} hand prompts at frames {sorted({p['frame_index'] for p in prompts})}")