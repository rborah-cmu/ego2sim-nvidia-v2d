import json, sys
from pathlib import Path
import cv2, numpy as np
ep, t = sys.argv[1], sys.argv[2]
E = Path.home() / f"v2d/ego2sim/ep{ep}"
for i in range(int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])):
    st = f"{i:06d}"
    m = cv2.imread(str(E / f"hand_masks/{t}/{st}.png"), 0)
    px = int((m > 0).sum()) if m is not None else -1
    out = f"{i} mask_px {px}"
    if px > 0:
        ys, xs = np.nonzero(m > 0); mb = (xs.min(), ys.min(), xs.max(), ys.max()); out += f" mbox {mb}"
    raw = json.load(open(E / f"hands/wilor_raw/{st}.json"))
    for d in raw:
        b = d["bbox"]
        iou = 0
        if px > 0:
            ix = max(0, min(mb[2], b["x1"]) - max(mb[0], b["x0"])); iy = max(0, min(mb[3], b["y1"]) - max(mb[1], b["y0"]))
            u = (mb[2]-mb[0])*(mb[3]-mb[1]) + (b["x1"]-b["x0"])*(b["y1"]-b["y0"]) - ix*iy
            iou = ix*iy/u if u else 0
        out += f" | det {'R' if d['is_right'] else 'L'} box({b['x0']:.0f},{b['y0']:.0f},{b['x1']:.0f},{b['y1']:.0f}) iou {iou:.2f}"
    mt = E / f"hands/wilor/{t}/{st}.json"
    out += f" | matched {mt.exists()} aligned {(E / f'hands/wilor_aligned/{t}/{st}.json').exists()}"
    print(out)