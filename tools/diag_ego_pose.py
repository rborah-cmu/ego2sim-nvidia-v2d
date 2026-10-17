"""Diagnose ego calibration conventions: estimate the cam_b -> cam_c relative pose from SIFT matches
(essential matrix on points undistorted with each intrinsics hypothesis) and compare with the calibration.

Usage: python diag_ego_pose.py CALIB_JSON VIDEO_B VIDEO_C [frame ...]
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ego_calib import load_cams  # noqa: E402


def frame(path, i):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    return cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)


def matches(A, B):
    sift = cv2.SIFT_create(6000)
    k1, d1 = sift.detectAndCompute(A, None)
    k2, d2 = sift.detectAndCompute(B, None)
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < 0.7 * b.distance]
    return (np.float64([k1[g.queryIdx].pt for g in good]), np.float64([k2[g.trainIdx].pt for g in good]))


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    calib, vb, vc = sys.argv[1:4]
    frames = [int(f) for f in sys.argv[4:]] or [0, 90]
    pb, pc = [], []
    for f in frames:
        a, b = matches(frame(vb, f), frame(vc, f))
        pb.append(a)
        pc.append(b)
    pb, pc = np.vstack(pb), np.vstack(pc)
    print(f"{len(pb)} matches; raw pixel median dx(c-b) {np.median(pc[:, 0] - pb[:, 0]):.1f}  dy {np.median(pc[:, 1] - pb[:, 1]):.1f}")
    for rotated in (True, False):
        for zero_tilt in (False, True):
            cams = load_cams(calib, rotated=rotated)
            B, C = cams["ego_cam_b"], cams["ego_cam_c"]
            db, dc = B.dist.copy(), C.dist.copy()
            if zero_tilt:
                db[12:] = 0
                dc[12:] = 0
            nb = cv2.undistortPoints(pb.reshape(-1, 1, 2), B.K, db).reshape(-1, 2)
            nc = cv2.undistortPoints(pc.reshape(-1, 1, 2), C.K, dc).reshape(-1, 2)
            E, inl = cv2.findEssentialMat(nb, nc, np.eye(3), cv2.RANSAC, 0.999, 1.5 / 560)
            _, R, t, _ = cv2.recoverPose(E, nb, nc, np.eye(3), mask=inl)
            T_cal = C.T_from_b  # b frame -> c frame
            t_cal = T_cal[:3, 3] / np.linalg.norm(T_cal[:3, 3])
            # epipolar residual of the calibrated pose on these points
            tx = np.array([[0, -t_cal[2], t_cal[1]], [t_cal[2], 0, -t_cal[0]], [-t_cal[1], t_cal[0], 0]])
            Ec = tx @ T_cal[:3, :3]
            hb = np.c_[nb, np.ones(len(nb))]
            hc = np.c_[nc, np.ones(len(nc))]
            l = hb @ Ec.T
            res = np.abs((hc * l).sum(1)) / np.linalg.norm(l[:, :2], axis=1) * 560
            print(f"rotated={rotated!s:5} zero_tilt={zero_tilt!s:5}: inliers {int(inl.sum()):5d}  est t {np.round(t.ravel(), 3)}  "
                  f"cal t {np.round(t_cal, 3)}  est R {rot_deg(R):.2f}deg  cal R {rot_deg(T_cal[:3, :3]):.2f}deg  "
                  f"R diff {rot_deg(R.T @ T_cal[:3, :3]):.2f}deg  calib epipolar residual median {np.median(res):.2f}px")


if __name__ == "__main__":
    main()
