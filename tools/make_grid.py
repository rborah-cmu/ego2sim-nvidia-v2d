"""2x2 sheet of frames (0, n/3, 2n/3, last) of a video, each scaled to 1000 px wide with a labelled 100-px grid.
Full-res coords = displayed x SCALE (printed). make_grid.py VIDEO OUT.jpg [--frames a,b,c,d]"""
import argparse
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--frames", default="")
a = ap.parse_args()
cap = cv2.VideoCapture(a.video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fr = [int(x) for x in a.frames.split(",")] if a.frames else [0, n // 3, 2 * n // 3, n - 1]
tiles = []
for f in fr:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    s = 1000 / im.shape[1]; im = cv2.resize(im, (1000, int(im.shape[0] * s)))
    for x in range(0, 1000, 100):
        cv2.line(im, (x, 0), (x, im.shape[0]), (0, 255, 255), 1); cv2.putText(im, str(x), (x + 2, 14), 0, 0.45, (0, 255, 255), 1)
    for y in range(0, im.shape[0], 100):
        cv2.line(im, (0, y), (1000, y), (0, 255, 255), 1); cv2.putText(im, str(y), (2, y + 14), 0, 0.45, (0, 255, 255), 1)
    cv2.putText(im, f"frame {f}", (420, 40), 0, 1.0, (0, 0, 255), 2); tiles.append(im)
sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"{a.out}: frames {fr} of {n}, scale {1 / s:.4f}")