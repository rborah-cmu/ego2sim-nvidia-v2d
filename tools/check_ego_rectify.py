"""Test calibration conventions: rectify the ego stereo pair under both hypotheses (calibration in rotated
image coords vs raw sensor coords) and report the vertical misalignment of SIFT matches.

Usage: python check_ego_rectify.py CALIB_JSON VIDEO_B VIDEO_C [frame ...]
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ego_calib import load_cams, rectify_pair  # noqa: E402


def frame(path, i):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def dy_stats(L, R):
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(L, None)
    k2, d2 = sift.detectAndCompute(R, None)
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < 0.7 * b.distance]
    p1 = np.float32([k1[g.queryIdx].pt for g in good])
    p2 = np.float32([k2[g.trainIdx].pt for g in good])
    dy = p1[:, 1] - p2[:, 1]
    dx = p1[:, 0] - p2[:, 0]
    return len(good), np.median(np.abs(dy)), np.mean(np.abs(dy) < 1.0), np.median(dx)


def main():
    calib, vb, vc = sys.argv[1:4]
    frames = [int(f) for f in sys.argv[4:]] or [0, 90]
    for rotated in (True, False):
        cams = load_cams(calib, rotated=rotated)
        pairs = [("left=c,right=b", cams["ego_cam_c"], cams["ego_cam_b"], vc, vb),
                 ("left=b,right=c", cams["ego_cam_b"], cams["ego_cam_c"], vb, vc)]
        for label, L, Rc, vl, vr in pairs:
            m1, m2, K, base, _ = rectify_pair(L, Rc)
            res = []
            for f in frames:
                rl = cv2.remap(frame(vl, f), *m1, cv2.INTER_LINEAR)
                rr = cv2.remap(frame(vr, f), *m2, cv2.INTER_LINEAR)
                res.append(dy_stats(rl, rr))
            n, mdy, frac, mdx = np.mean(res, axis=0)
            print(f"calib_rotated={rotated!s:5} {label}: matches {n:5.0f}  median|dy| {mdy:6.2f}px  "
                  f"|dy|<1px {frac:5.1%}  median dx {mdx:7.1f}px  baseline {base*100:.2f} cm  f {K[0,0]:.1f}")


if __name__ == "__main__":
    main()
