"""Compare a FoundationPose object-object relative trajectory against Track 3 mocap.

Needs no camera->world extrinsic: for a target object A and a reference object B (ideally static),
  est  B_T_A(t) = inv(cam_T_B(t)) @ cam_T_A(t)      (FoundationPose JSONs, wxyz rotation)
  gt   B_T_A(t) = inv(world_T_B(t)) @ world_T_A(t)  (parquet poses, [x,y,z,qw,qx,qy,qz])

Usage: python compare_relpose.py PARQUET POSES_A NAME_A POSES_B NAME_B [--axis-symmetric-a]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


def mat(t, q_wxyz):
    m = np.eye(4)
    m[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    m[:3, 3] = t
    return m


def load_fp(d: Path, n: int):
    out = np.full((n, 4, 4), np.nan)
    for i in range(n):
        p = d / f"{i:06d}.json"
        if p.exists():
            j = json.loads(p.read_text())
            out[i] = mat(j["translation"], j["rotation"])
    return out


def load_gt(parquet: str, name: str):
    rows = pq.read_table(parquet, columns=["observation.objects"]).column(0).to_pylist()
    out = np.full((len(rows), 4, 4), np.nan)
    for i, objs in enumerate(rows):
        o = next(o for o in objs if o["name"] == name)
        if o["visible"]:
            out[i] = mat(o["pose"][:3], o["pose"][3:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("poses_a")
    ap.add_argument("name_a")
    ap.add_argument("poses_b")
    ap.add_argument("name_b")
    ap.add_argument("--axis-symmetric-a", action="store_true",
                    help="object A is rotationally symmetric about its local z axis (cups): compare axis tilt only")
    args = ap.parse_args()

    gt_a, gt_b = load_gt(args.parquet, args.name_a), load_gt(args.parquet, args.name_b)
    n = len(gt_a)
    fp_a, fp_b = load_fp(Path(args.poses_a), n), load_fp(Path(args.poses_b), n)
    est = np.linalg.inv(fp_b) @ fp_a
    gt = np.linalg.inv(gt_b) @ gt_a
    ok = np.isfinite(est).all((1, 2)) & np.isfinite(gt).all((1, 2))

    pos_err = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=1) * 100
    if args.axis_symmetric_a:
        za, zg = est[:, :3, 2], gt[:, :3, 2]
        rot_err = np.degrees(np.arccos(np.clip((za * zg).sum(1), -1, 1)))
        rot_label = "axis tilt err (deg)"
    else:
        rel = np.einsum("nji,njk->nik", gt[:, :3, :3], est[:, :3, :3])
        rot_err = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        rot_label = "rot err (deg)"
    # B should be static in the world: its camera-frame motion vs. the static mocap shows FP jitter.
    gt_b_static = np.nanmax(np.linalg.norm(gt_b[:, :3, 3] - gt_b[0, :3, 3], axis=1)) * 100
    gt_travel = np.linalg.norm(gt[:, :3, 3] - gt[0, :3, 3], axis=1) * 100

    print(f"frames compared: {ok.sum()}/{n}   ({args.name_a} in {args.name_b} frame)")
    print(f"mocap: {args.name_b} moves at most {gt_b_static:.1f} cm; {args.name_a} travels up to "
          f"{np.nanmax(gt_travel[ok]):.1f} cm relative to it")
    for label, e in (("pos err (cm)", pos_err), (rot_label, rot_err)):
        v = e[ok]
        print(f"{label:22s} mean {v.mean():6.2f}  median {np.median(v):6.2f}  p90 {np.percentile(v, 90):6.2f}  max {v.max():6.2f}")
    print("per-segment mean pos err (cm):",
          " ".join(f"{pos_err[ok][s].mean():.1f}" for s in np.array_split(np.arange(ok.sum()), 6)))
    print(f"frame 0 relative position est {np.round(est[0, :3, 3] * 100, 1)} cm vs gt {np.round(gt[0, :3, 3] * 100, 1)} cm")


if __name__ == "__main__":
    main()
