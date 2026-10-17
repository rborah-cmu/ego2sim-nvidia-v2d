"""Estimate ego-rig extrinsics from the videos themselves (the factory extrinsics' conventions don't
reproduce a clean rectification on the published videos).

For a camera pair (X, Y) it pools SIFT matches over several frames, undistorts them with the factory
intrinsics (raw or rotated hypothesis), estimates the essential matrix and recovers R, t_dir (points
X frame -> Y frame). Metric scale comes from the factory baseline length |t|.
It prints per-hypothesis inlier counts and the pose spread across frame subsets, and writes
the chosen result to JSON.

Usage: python estimate_ego_extrinsics.py CALIB_JSON EPISODE_VIDEO_DIR EPISODE OUT_JSON
"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ego_calib import load_cams  # noqa: E402

FRAMES = [0, 40, 80, 120, 160]


def gray(path, i):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if ok else None


def matches(A, B):
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(A, None)
    k2, d2 = sift.detectAndCompute(B, None)
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < 0.7 * b.distance]
    return np.float64([k1[g.queryIdx].pt for g in good]), np.float64([k2[g.trainIdx].pt for g in good])


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def estimate(nx, ny, f_px):
    E, inl = cv2.findEssentialMat(nx, ny, np.eye(3), cv2.RANSAC, 0.9999, 1.0 / f_px)
    _, R, t, inl2 = cv2.recoverPose(E, nx, ny, np.eye(3), mask=inl)
    return R, t.ravel(), int((inl2 > 0).sum())


def pair(cams, vx, vy, cx, cy, ep, video_dir):
    px, py = [], []
    for f in FRAMES:
        a = gray(f"{video_dir}/observation.images.{vx}/episode_{ep:06d}.mp4", f)
        b = gray(f"{video_dir}/observation.images.{vy}/episode_{ep:06d}.mp4", f)
        if a is None or b is None:
            continue
        if a.shape != b.shape:  # cam_a is 2028x1520, mono cams 1280x800: match at similar scale
            pass
        p, q = matches(a, b)
        px.append(p)
        py.append(q)
    X, Y = cams[cx], cams[cy]
    nx = cv2.undistortPoints(np.vstack(px).reshape(-1, 1, 2), X.K, X.dist).reshape(-1, 2)
    ny = cv2.undistortPoints(np.vstack(py).reshape(-1, 1, 2), Y.K, Y.dist).reshape(-1, 2)
    f_px = min(X.K[0, 0], Y.K[0, 0])
    R, t, n = estimate(nx, ny, f_px)
    # stability: re-estimate on halves
    idx = np.random.default_rng(0).permutation(len(nx))
    spread = []
    for half in (idx[: len(idx) // 2], idx[len(idx) // 2:]):
        Rh, th, _ = estimate(nx[half], ny[half], f_px)
        spread.append((rot_deg(Rh.T @ R), float(np.degrees(np.arccos(np.clip(th @ t, -1, 1))))))
    return R, t, n, len(nx), spread


def main():
    calib, video_dir, ep, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    factory = load_cams(calib, rotated=False)
    base_bc = np.linalg.norm(factory["ego_cam_c"].T_from_b[:3, 3])
    base_ba = np.linalg.norm(factory["ego_cam_a"].T_from_b[:3, 3])
    result = {}
    for rotated in (False, True):
        cams = load_cams(calib, rotated=rotated)
        for vx, vy, cx, cy in (("ego_cam_b", "ego_cam_c", "ego_cam_b", "ego_cam_c"),
                               ("ego_cam_b", "ego_cam_c", "ego_cam_c", "ego_cam_b"),
                               ("ego_cam_b", "ego_cam_a", "ego_cam_b", "ego_cam_a"),
                               ("ego_cam_c", "ego_cam_a", "ego_cam_c", "ego_cam_a")):
            R, t, n, tot, spread = pair(cams, vx, vy, cx, cy, ep, video_dir)
            key = f"{'rot' if rotated else 'raw'}:{vx}->{vy}:K({cx},{cy})"
            print(f"{key:48s} inliers {n:5d}/{tot:5d}  R {rot_deg(R):5.2f}deg  t_dir {np.round(t, 3)}  "
                  f"half-split spread R {spread[0][0]:.2f}/{spread[1][0]:.2f}deg t {spread[0][1]:.1f}/{spread[1][1]:.1f}deg")
            result[key] = {"inliers": n, "matches": tot, "R": R.tolist(), "t_dir": t.tolist(),
                           "baseline_m": float(base_ba if vy == "ego_cam_a" else base_bc)}
    json.dump(result, open(out, "w"), indent=1)
    print("factory baselines: b-c %.4f m, b-a %.4f m" % (base_bc, base_ba))


if __name__ == "__main__":
    main()
