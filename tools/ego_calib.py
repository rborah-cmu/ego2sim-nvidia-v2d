"""Ego-rig calibration helpers for the Track 3 videos.

The OAK factory calibration (meta/camera_calibration.json) is in RAW sensor coordinates, but the published ego
videos are rotated 180 deg (device rotation_deg=180). For a 180 deg in-plane image rotation:
  * principal point: cx' = W-1-cx, cy' = H-1-cy
  * distortion: tangential p1, p2 and the thin-prism / tilt terms change sign (radial terms unchanged)
  * extrinsics are conjugated by F = diag(-1,-1,1): R' = F R F, t' = F t
OAK `extrinsics_to_stereo_left[cam]` maps points from the raw-left camera (cam_b) frame into `cam`'s frame
(cam_c has t_x = -7.5 cm). The rotated ego pair is left=cam_c, right=cam_b.
"""
import json
from dataclasses import dataclass

import numpy as np

F3 = np.diag([-1.0, -1.0, 1.0])


@dataclass
class Cam:
    name: str
    K: np.ndarray
    dist: np.ndarray
    size: tuple  # (W, H)
    T_from_b: np.ndarray  # 4x4: point in (raw or rotated) cam_b frame -> point in this camera's frame


def load_cams(calib_path: str, rotated: bool = True) -> dict:
    d = json.load(open(calib_path))["cameras"]
    cams = {}
    for name in ("ego_cam_a", "ego_cam_b", "ego_cam_c"):
        c = d[name]
        W, H = c["resolution"]
        K = np.array(c["intrinsic_matrix"], dtype=np.float64)
        dist = np.array(c["distortion_coefficients"], dtype=np.float64)
        T = np.array(c["extrinsics_to_stereo_left"], dtype=np.float64) if "extrinsics_to_stereo_left" in c else np.eye(4)
        T[:3, 3] /= 100.0 if np.abs(T[:3, 3]).max() > 1.0 else 1.0  # guard: OAK sometimes reports cm
        if rotated:
            K = K.copy()
            K[0, 2] = W - 1 - K[0, 2]
            K[1, 2] = H - 1 - K[1, 2]
            dist = dist.copy()
            # OpenCV 14-coef order: k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tx ty
            for i in (2, 3, 8, 9, 10, 11):
                dist[i] = -dist[i]
            T = T.copy()
            T[:3, :3] = F3 @ T[:3, :3] @ F3
            T[:3, 3] = F3 @ T[:3, 3]
        cams[name] = Cam(name, K, dist, (W, H), T)
    return cams


def rectify_pair(left: Cam, right: Cam, size=None, alpha=0.0):
    """cv2.stereoRectify for (left, right); returns maps + rectified K, baseline and R_left (rect<-left)."""
    import cv2
    T_l_r = right.T_from_b @ np.linalg.inv(left.T_from_b)  # left frame -> right frame
    R, t = T_l_r[:3, :3].copy(), T_l_r[:3, 3].reshape(3, 1).copy()
    size = size or left.size
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(left.K, left.dist, right.K, right.dist, size, R, t,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    m1 = cv2.initUndistortRectifyMap(left.K, left.dist, R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(right.K, right.dist, R2, P2, size, cv2.CV_32FC1)
    baseline = -P2[0, 3] / P2[0, 0]
    return m1, m2, P1[:3, :3], baseline, R1
