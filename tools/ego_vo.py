"""Stereo-depth visual odometry for the ego head camera (cam_a, undistorted pinhole) -> camera_to_world per frame.

World = camera frame 0 (the convention of the reconstruction bundle's result.npz). Static scene only: hands and every
tracked object are masked out. Per frame t, SIFT features are matched to the current KEYFRAME, whose features carry
metric 3D points from our stereo depth; PnP-RANSAC (+ LM refinement on the inliers) gives the camera pose relative to
the keyframe. A new keyframe is taken when inliers drop below --min-inliers, which bounds drift between keyframes.

Usage:
  python ego_vo.py STEREO_DIR OUT_NPZ --masks DIR[,DIR...] [--scale 0.5] [--min-inliers 200]
  STEREO_DIR holds cam_a_undist.mp4, stereo_depth/*.png (v2d inverse depth), intrinsics_undist.json
  (from tools/ego_stereo_depth.py); --masks are per-object/per-hand dirs of NNNNNN.png to exclude.
Output npz: camera_to_world (N,4,4) float64, inliers (N,), keyframe (N,) = keyframe index used, K (3,3) full-res.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def decode_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[(raw >= 65535) | (raw <= 0)] = 0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stereo_dir")
    ap.add_argument("out_npz")
    ap.add_argument("--masks", default="")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--min-inliers", type=int, default=200)
    ap.add_argument("--max-depth", type=float, default=4.0)
    a = ap.parse_args()
    S = Path(a.stereo_dir)
    Kj = json.load(open(S / "intrinsics_undist.json"))
    K = np.array([[Kj["fx"], 0, Kj["cx"]], [0, Kj["fy"], Kj["cy"]], [0, 0, 1]], float)
    Ks = K.copy()
    Ks[:2] *= a.scale
    mask_dirs = [Path(m) for m in a.masks.split(",") if m]
    sift = cv2.SIFT_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    dil = np.ones((31, 31), np.uint8)

    cap = cv2.VideoCapture(str(S / "cam_a_undist.mp4"))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    N = len(frames)

    def features(i):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        dyn = np.zeros(g.shape, np.uint8)
        for md in mask_dirs:
            m = cv2.imread(str(md / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                dyn |= (m > 0).astype(np.uint8)
        keep = (cv2.dilate(dyn, dil) == 0).astype(np.uint8) * 255
        gs = cv2.resize(g, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        ks = cv2.resize(keep, (gs.shape[1], gs.shape[0]), interpolation=cv2.INTER_NEAREST)
        kp, des = sift.detectAndCompute(gs, ks)
        uv = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        return uv, des

    def lift(i, uv):
        """3D points (camera i frame, metres) for scaled pixel coords uv; NaN where no stereo depth."""
        d = decode_depth(S / "stereo_depth" / f"{i:06d}.png")
        full = uv / a.scale
        x = np.clip(np.round(full[:, 0]).astype(int), 0, d.shape[1] - 1)
        y = np.clip(np.round(full[:, 1]).astype(int), 0, d.shape[0] - 1)
        # robust: median of the 5x5 neighbourhood's valid depths (stereo has holes and edge noise)
        z = np.full(len(uv), np.nan)
        for j, (xx, yy) in enumerate(zip(x, y)):
            w = d[max(yy - 2, 0):yy + 3, max(xx - 2, 0):xx + 3]
            w = w[(w > 0.1) & (w < a.max_depth)]
            if w.size >= 5 and (w.max() - w.min()) < 0.05 * np.median(w):
                z[j] = np.median(w)
        X = np.stack([(full[:, 0] - K[0, 2]) * z / K[0, 0], (full[:, 1] - K[1, 2]) * z / K[1, 1], z], 1)
        return X

    def make_kf(i, c2w):
        uv, des = features(i)
        X = lift(i, uv)
        ok = np.isfinite(X[:, 2])
        return {"i": i, "uv": uv[ok], "des": des[ok], "X": X[ok], "c2w": c2w}

    def solve(kf, i):
        uv, des = features(i)
        if des is None or len(des) < 10:
            return None, 0
        m = bf.knnMatch(kf["des"], des, k=2)
        good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 12:
            return None, len(good)
        obj = np.float64([kf["X"][g.queryIdx] for g in good])
        img = np.float64([uv[g.trainIdx] for g in good])
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, Ks, None, iterationsCount=2000, reprojectionError=2.0,
                                                     confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:  # degenerate sample (e.g. near-collinear points): treat as a failed match
            return None, 0
        if not ok or inl is None or len(inl) < 12:
            return None, 0 if inl is None else len(inl)
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], Ks, None, rvec, tvec)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec[:, 0]  # X_cam_i = T @ X_cam_kf
        return T, len(inl)

    c2w = np.zeros((N, 4, 4))
    c2w[0] = np.eye(4)
    inliers = np.zeros(N, int)
    kfidx = np.zeros(N, int)
    kf = make_kf(0, np.eye(4))
    for i in range(1, N):
        T, n = solve(kf, i)
        if T is None or n < a.min_inliers:
            if kf["i"] != i - 1:  # re-key on the previous frame and retry
                kf = make_kf(i - 1, c2w[i - 1])
                T, n = solve(kf, i)
            if T is None:  # tracking lost: hold the previous pose (logged)
                print(f"frame {i}: tracking lost ({n} inliers), holding pose")
                c2w[i] = c2w[i - 1]
                inliers[i], kfidx[i] = n, kf["i"]
                continue
        c2w[i] = kf["c2w"] @ np.linalg.inv(T)
        inliers[i], kfidx[i] = n, kf["i"]
    np.savez(a.out_npz, camera_to_world=c2w, inliers=inliers, keyframe=kfidx, K=K)
    path = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum()
    print(f"{N} frames, {len(set(kfidx.tolist()))} keyframes, inliers median {int(np.median(inliers[1:]))} "
          f"min {inliers[1:].min()}, camera path {path*100:.1f} cm -> {a.out_npz}")


if __name__ == "__main__":
    main()
