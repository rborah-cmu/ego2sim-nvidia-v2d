"""Per-frame world wrist of a hand-track dir + source track + gap flags. trace_hand.py EP SIDE [FROM TO]"""
import json, sys
from pathlib import Path
import numpy as np
ep, side = sys.argv[1], sys.argv[2]
E = Path.home() / f"v2d/ego2sim/ep{ep}"
c2w = np.load(E / "camera_vo.npz")["camera_to_world"]
sid = {"left": "0", "right": "1"}[side]
al = E / "hands/wilor_aligned"
tracks = [d for d in sorted(al.iterdir()) if d.is_dir()]
def wrist(j):
    for k in ("cam_t", "pred_cam_t_full", "transl", "wrist_cam"):
        if k in j:
            return np.asarray(j[k], float).reshape(-1)[:3]
    return None
n = len(list((E / "hands/frames").glob("*.jpg")))
f0, f1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, n)
first = None
for i in range(f0, f1):
    st = f"{i:06d}"
    src = [t.name for t in tracks if (t / f"{st}.json").exists()]
    p = E / f"hands/side_aligned/{sid}/{st}.json"
    c = E / f"hands/clean/{side}/{st}.json"
    pick = "-"
    if p.exists():
        txt = p.read_text()
        pick = next((t.name for t in tracks if (t / f"{st}.json").exists() and (t / f"{st}.json").read_text() == txt), "?")
    w = "missing"
    if c.exists():
        j = json.load(open(c))
        if first is None:
            first = sorted(j.keys())
        x = wrist(j)
        if x is not None:
            T = c2w[i]
            w = np.round(T[:3, :3] @ x + T[:3, 3], 3).tolist()
    print(i, "aligned_in", src, "side_pick", pick, "clean", "cleaned" if c.exists() and json.load(open(c)).get("cleaned") else "", w)
print("keys", first)