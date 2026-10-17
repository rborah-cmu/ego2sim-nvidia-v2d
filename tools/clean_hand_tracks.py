"""Repair glitches in per-frame MANO hand tracks (wilor_aligned_filled/<track>/<frame>.json) before the bundle.

Residual wrist jumps (hands entering/leaving the view, extrapolated frames, wrong WiLoR matches) make the reference wrist
move 10-60 cm in one frame; the policy cannot follow, episodes end on wrist termination, and the last curriculum stage
(50% resets to frame 0, where the hands enter the view) then wrecks the learned grasp (ep12, Phase 5f).

Works in WORLD coordinates (camera_to_world from tools/ego_vo.py) so head motion is not mistaken for hand motion:
  wrist_w = R_c2w @ cam_t + t_c2w,   R_w = R_c2w @ R(global_orient)
A frame is BAD if its wrist is > max(--pos-k * MAD, --pos-min) from the median of a centred window, or its orientation
is > --rot-max deg from the window's mean rotation, or it moves faster than --vmax (m/s) relative to BOTH neighbours.
Bad frames are re-filled from the nearest good frames (position lerp, orientation slerp in world, finger pose and betas
lerp; held at the ends) and converted back to the camera frame. Written as the same JSON layout with "cleaned": true.

Usage: python clean_hand_tracks.py CAM_NPZ IN_DIR OUT_DIR [--fps 20.25] [--win 7] [--vmax 1.5]   (IN/OUT = one track dir)
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam_npz")
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=20.25)
    ap.add_argument("--win", type=int, default=7)
    ap.add_argument("--pos-k", type=float, default=4.0)
    ap.add_argument("--pos-min", type=float, default=0.03)
    ap.add_argument("--rot-max", type=float, default=45.0)
    ap.add_argument("--vmax", type=float, default=1.5)
    a = ap.parse_args()
    c2w = np.load(a.cam_npz)["camera_to_world"]
    files = sorted(Path(a.in_dir).glob("*.json"))
    if len(files) < 2:  # hand never reconstructed (e.g. idle, cut off at the image edge): nothing to clean
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        for f in files:
            Path(a.out_dir, f.name).write_text(f.read_text())
        print(f"{a.in_dir}: {len(files)} frames, nothing to repair")
        return
    idx = np.array([int(f.stem) for f in files])
    recs = [json.load(open(f)) for f in files]
    n = len(recs)
    t_c = np.array([r["camera"]["cam_t"] for r in recs], float)
    o_c = np.array([r["mano"]["global_orient"] for r in recs], float).reshape(n, 3)
    Rw = R.from_matrix(np.einsum("nij,njk->nik", c2w[idx, :3, :3], R.from_rotvec(o_c).as_matrix()))
    pw = np.einsum("nij,nj->ni", c2w[idx, :3, :3], t_c) + c2w[idx, :3, 3]

    h = a.win // 2
    bad = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        nb = [j for j in range(lo, hi) if j != i]
        med = np.median(pw[nb], axis=0)
        mad = np.median(np.linalg.norm(pw[nb] - med, axis=1)) + 1e-4
        if np.linalg.norm(pw[i] - med) > max(a.pos_k * mad, a.pos_min):
            bad[i] = True
        if np.degrees((Rw[nb].mean() * Rw[i].inv()).magnitude()) > a.rot_max:
            bad[i] = True
    step = np.linalg.norm(np.diff(pw, axis=0), axis=1) * a.fps
    for i in range(1, n - 1):
        if step[i - 1] > a.vmax and step[i] > a.vmax:
            bad[i] = True
    good = np.nonzero(~bad)[0]
    if len(good) < 2:
        raise SystemExit("too few good frames to repair")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slerp = Slerp(good.astype(float), Rw[good])
    fp = np.array([np.asarray(r["mano"]["hand_pose"], float).reshape(45) for r in recs])
    for i in range(n):
        rec = copy.deepcopy(recs[i])
        if bad[i]:
            ii = float(np.clip(i, good[0], good[-1]))
            p = np.array([np.interp(ii, good, pw[good, k]) for k in range(3)])
            Rwi = slerp([ii])[0]
            fpi = np.array([np.interp(ii, good, fp[good, k]) for k in range(45)])
            Rc = c2w[idx[i], :3, :3]
            rec["camera"]["cam_t"] = (Rc.T @ (p - c2w[idx[i], :3, 3])).tolist()
            rec["mano"]["global_orient"] = R.from_matrix(Rc.T @ Rwi.as_matrix()).as_rotvec().tolist()
            rec["mano"]["hand_pose"] = fpi.reshape(15, 3).tolist()
            rec["cleaned"] = True
        json.dump(rec, open(out / files[i].name, "w"))
    runs, s = [], None
    for i in range(n + 1):
        if i < n and bad[i]:
            s = i if s is None else s
        elif s is not None:
            runs.append(f"{idx[s]}-{idx[i - 1]}" if i - 1 > s else f"{idx[s]}")
            s = None
    before = np.linalg.norm(np.diff(pw, axis=0), axis=1).max() * 100
    print(f"{a.in_dir}: repaired {bad.sum()}/{n} frames [{', '.join(runs)}]; max world wrist step before {before:.1f} cm")


if __name__ == "__main__":
    main()
