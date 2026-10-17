"""Mask overlay sheet: 4 frames of cam_a with object masks (colour per id) + hand masks (union, grey) overlaid.
  mask_sheet.py EPDIR OUT.jpg"""
import sys
from pathlib import Path
import cv2, numpy as np
E = Path(sys.argv[1]); out = sys.argv[2]
cap = cv2.VideoCapture(str(E / "stereo/cam_a_undist.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
tiles = []
for f in [0, n // 3, 2 * n // 3, n - 1]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); im = im.astype(float)
    for d in sorted((E / "hand_masks").glob("*")):
        m = cv2.imread(str(d / f"{f:06d}.png"), 0)
        if m is not None:
            im[m > 0] = 0.5 * im[m > 0] + 0.5 * np.array([200, 200, 200])
    for i, d in enumerate(sorted((E / "masks").glob("*"), key=lambda p: int(p.name))):
        m = cv2.imread(str(d / f"{f:06d}.png"), 0)
        if m is not None:
            im[m > 0] = 0.45 * im[m > 0] + 0.55 * np.array(cols[i % 4])
    im = cv2.resize(im.astype(np.uint8), (1000, 750)); cv2.putText(im, f"frame {f}", (420, 40), 0, 1.0, (255, 255, 255), 2)
    tiles.append(im)
cv2.imwrite(out, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(out)