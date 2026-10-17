"""World-frame object tracks vs mocap on a public Track 3 episode (local proxy for the Kaggle pose metrics).

est world pose of object o at frame t:  W_T_o(t) = camera_to_world(t) @ cam_T_o(t)   (FoundationPose, scan mesh frame)
Like Kaggle, ONE rigid world transform is fitted from the initial scene (all objects' scan vertices at frame 0) and
applied to every frame; nothing is rescaled. Also reports a best-fit-over-the-clip alignment, and a no-camera-motion
control (camera_to_world = I) to show what the odometry buys.

Usage:
  python validate_world_tracks.py PARQUET CAM_NPZ|none NAME=FP_POSE_DIR [NAME=FP_POSE_DIR ...] [--mesh NAME=PATH]
Prints per object: centre error, ADD (mean vertex distance), rotation error (mean / median / p90), and the fraction of
frames with centre < 3 cm & rot < 30 deg (the MP-SR per-object condition).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

SCANS = Path.home() / "v2d/v2d_data/track_3/public/mesh"


def fp_track(d, n):
    out = np.full((n, 4, 4), np.nan)
    for i in range(n):
        p = Path(d) / f"{i:06d}.json"
        if p.exists():
            j = json.load(open(p))
            w, x, y, z = j["rotation"]
            out[i] = np.eye(4)
            out[i][:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
            out[i][:3, 3] = j["translation"]
    return out


def gt_track(parquet, name):
    rows = pq.read_table(parquet, columns=["observation.objects"]).column(0).to_pylist()
    out = np.zeros((len(rows), 4, 4))
    for i, objs in enumerate(rows):
        o = next(o for o in objs if o["name"] == name)
        x, y, z, qw, qx, qy, qz = o["pose"]
        out[i] = np.eye(4)
        out[i][:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        out[i][:3, 3] = [x, y, z]
    return out


def umeyama_rigid(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ D @ Vt
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = md - Rm @ ms
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def evaluate(c2w, objs, verts, gts, label):
    est = {o: np.einsum("nij,njk->nik", c2w, objs[o]) for o in objs}
    n = len(c2w)
    ok0 = all(np.isfinite(est[o][0]).all() for o in objs)
    assert ok0, "frame 0 must be tracked for every object"
    # one rigid transform from the initial scene (all objects' vertices at frame 0)
    src = np.vstack([apply(est[o][0], verts[o]) for o in objs])
    dst = np.vstack([apply(gts[o][0], verts[o]) for o in objs])
    W0 = umeyama_rigid(src, dst)
    # best-fit over the clip (centres of every object, every frame)
    srcs, dsts = [], []
    for o in objs:
        c = verts[o].mean(0)
        for t in range(n):
            if np.isfinite(est[o][t]).all():
                srcs.append(apply(est[o][t], c[None])[0]); dsts.append(apply(gts[o][t], c[None])[0])
    Wb = umeyama_rigid(np.array(srcs), np.array(dsts))
    print(f"== {label}")
    for wl, W in (("initial-scene alignment", W0), ("best-fit alignment", Wb)):
        for o in objs:
            c = verts[o].mean(0)
            ce, add, rot, both = [], [], [], []
            for t in range(n):
                if not np.isfinite(est[o][t]).all():
                    continue
                E = W @ est[o][t]
                G = gts[o][t]
                ce.append(np.linalg.norm(apply(E, c[None]) - apply(G, c[None])) * 100)
                add.append(np.linalg.norm(apply(E, verts[o]) - apply(G, verts[o]), axis=1).mean() * 100)
                rot.append(np.degrees((R.from_matrix(E[:3, :3]) * R.from_matrix(G[:3, :3]).inv()).magnitude()))
                both.append(ce[-1] < 3 and rot[-1] < 30)
            ce, add, rot = map(np.array, (ce, add, rot))
            print(f"  {wl:23s} {o:18s} centre {ce.mean():5.2f}/{np.median(ce):5.2f}/{np.percentile(ce,90):5.2f} cm | "
                  f"ADD {add.mean():5.2f} cm | rot {rot.mean():5.1f}/{np.median(rot):5.1f}/{np.percentile(rot,90):5.1f} deg | "
                  f"<3cm&<30deg {np.mean(both):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("cam_npz")
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--mesh", action="append", default=[])
    a = ap.parse_args()
    n = pq.read_metadata(a.parquet).num_rows
    meshes = dict(m.split("=", 1) for m in a.mesh)
    objs, verts, gts = {}, {}, {}
    for spec in a.tracks:
        name, d = spec.split("=", 1)
        objs[name] = fp_track(d, n)
        mp = meshes.get(name, SCANS / name / f"{name}.glb")
        mp = mp if Path(mp).exists() else SCANS / name / f"{name}_visual.glb"
        m = trimesh.load(str(mp), force="mesh")
        verts[name] = np.asarray(trimesh.sample.sample_surface(m, 2000, seed=0)[0])
        gts[name] = gt_track(a.parquet, name)
        gtrav = np.linalg.norm(gts[name][:, :3, 3] - gts[name][0, :3, 3], axis=1).max() * 100
        print(f"{name}: {np.isfinite(objs[name][:, 0, 0]).sum()}/{n} tracked frames, mocap travel {gtrav:.1f} cm")
    if a.cam_npz != "none":
        c2w = np.load(a.cam_npz)["camera_to_world"]
        assert len(c2w) == n, f"camera frames {len(c2w)} != parquet rows {n}"
        evaluate(c2w, objs, verts, gts, f"WITH camera motion ({Path(a.cam_npz).name})")
    evaluate(np.repeat(np.eye(4)[None], n, 0), objs, verts, gts, "control: NO camera motion (camera_to_world = I)")


if __name__ == "__main__":
    main()
