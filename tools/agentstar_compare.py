"""Compare AgentSTAR outputs (and FoundationPose runs using AgentSTAR meshes) with the public scans + mocap.

Meshes from AgentSTAR live in their own canonical frame (+Z up, ~unit size, SCALE in pose.json), so nothing
here assumes the agent's object frames match the scans' frames.

  mesh  AGENT_GLB AGENT_POSE_JSON SCAN_GLB
        size ratio (agent * SCALE vs scan, bbox diagonal) and symmetric Chamfer (cm) after a Sim(3) fit
        (multi-start ICP with scale; Kaggle's CD-O also registers each mesh with Sim(3)).
  traj  PARQUET NAME_A NAME_B SRC_A SRC_B [MESH_A MESH_B]
        Object A's mesh centre expressed in object B's frame, over time, vs mocap — fitted with ONE rigid
        transform over the whole clip (absorbs the unknown canonical-frame offset of B). Reports the per-frame
        residual in cm (and the Sim(3)-fitted residual, to show scale error). SRC_* is either an AgentSTAR
        pose.json (keyframes only) or a FoundationPose poses dir (JSON per frame, wxyz). MESH_* (optional,
        default: scans for FP dirs / agent glb next to pose.json) give each object's centre in its own frame.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

SCANS = Path.home() / "v2d/v2d_data/track_3/public/mesh"


def load_mesh(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    return np.asarray(m.vertices, float), m


def sample(m, n=20000, seed=0):
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts)


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ D @ Vt
    s = (S * np.diag(D)).sum() / xs.var(0).sum() if with_scale else 1.0
    return s, Rm, mu_d - s * Rm @ mu_s


def icp_sim3(src, dst, init_R, iters=40):
    tree = cKDTree(dst)
    s = np.sqrt(dst.var(0).sum() / src.var(0).sum())
    Rm, t = init_R, dst.mean(0) - s * init_R @ src.mean(0)
    for _ in range(iters):
        cur = (s * (Rm @ src.T)).T + t
        _, idx = tree.query(cur)
        s, Rm, t = umeyama(src, dst[idx])
    cur = (s * (Rm @ src.T)).T + t
    d1, _ = tree.query(cur)
    d2, _ = cKDTree(cur).query(dst)
    return (d1.mean() + d2.mean()) / 2, s, Rm, t


def cmd_mesh(agent_glb, agent_pose, scan_glb):
    scale = json.load(open(agent_pose)).get("scale", 1.0)
    _, am = load_mesh(agent_glb)
    _, sm = load_mesh(scan_glb)
    a = sample(am) * scale
    b = sample(sm)
    diag = lambda p: np.linalg.norm(p.max(0) - p.min(0))
    print(f"agent SCALE {scale:.4f}; bbox diag agent {diag(a)*100:.1f} cm vs scan {diag(b)*100:.1f} cm "
          f"(ratio {diag(a)/diag(b):.3f})")
    best = None
    for rv in R.create_group("O").as_matrix():  # 24 axis-aligned starts
        cd, s, Rm, t = icp_sim3(a[::4], b[::4], rv)
        if best is None or cd < best[0]:
            best = (cd, s, Rm, t)
    cd, s, Rm, t = icp_sim3(a, b, best[2])
    print(f"Chamfer after Sim(3) fit: {cd*100:.2f} cm (fit scale {s:.3f}; scan diag {diag(b)*100:.1f} cm)")


def mesh_centre(path):
    v, _ = load_mesh(path)
    return (v.min(0) + v.max(0)) / 2


def load_track(src, n_frames):
    """-> (frames idx array, (N,4,4) object->camera), plus the default mesh path for the centre."""
    src = Path(src)
    if src.is_file():  # AgentSTAR pose.json
        doc = json.load(open(src))
        s_doc = doc.get("scale", 1.0)
        idx, mats = [], []
        for name, fr in doc["frames"].items():
            op = fr["object_pose"]
            q = op["quaternion"]
            s = op.get("scale", s_doc)
            m = np.eye(4)
            m[:3, :3] = s * R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            m[:3, 3] = op["translation"]
            idx.append(int(Path(name).stem))
            mats.append(m)
        o = np.argsort(idx)
        return np.array(idx)[o], np.array(mats)[o], src.parent / "object.glb"
    idx, mats = [], []
    for i in range(n_frames):
        p = src / f"{i:06d}.json"
        if p.exists():
            j = json.load(open(p))
            w, x, y, z = j["rotation"]
            m = np.eye(4)
            m[:3, :3] = R.from_quat([x, y, z, w]).as_matrix() * np.array(j.get("scale", [1, 1, 1]))[None, :]
            m[:3, 3] = j["translation"]
            idx.append(i)
            mats.append(m)
    return np.array(idx), np.array(mats), None


def gt_poses(parquet, name):
    rows = pq.read_table(parquet, columns=["observation.objects"]).column(0).to_pylist()
    out = []
    for objs in rows:
        o = next(o for o in objs if o["name"] == name)
        m = np.eye(4)
        x, y, z, qw, qx, qy, qz = o["pose"]
        m[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        m[:3, 3] = [x, y, z]
        out.append(m)
    return np.array(out)


def cmd_traj(parquet, name_a, name_b, src_a, src_b, mesh_a=None, mesh_b=None):
    n = pq.read_metadata(parquet).num_rows
    ia, Ta, dma = load_track(src_a, n)
    ib, Tb, dmb = load_track(src_b, n)
    ca = mesh_centre(mesh_a or dma or SCANS / name_a / f"{name_a}.glb")
    if dma is not None and mesh_a is None:
        # AgentSTAR pose.json is in Blender's +Z-up canonical frame, but object.glb is glTF (+Y up):
        # glTF (x, y, z) = Blender (x, z, -y)  =>  Blender (x, y, z) = glTF (x, -z, y)
        ca = np.array([ca[0], -ca[2], ca[1]])
    common = np.intersect1d(ia, ib)
    Ta = Ta[np.searchsorted(ia, common)]
    Tb = Tb[np.searchsorted(ib, common)]
    # A's centre in B's frame (B's scale divided out so the B frame is rigid)
    est = []
    for ta, tb in zip(Ta, Tb):
        sb = np.cbrt(abs(np.linalg.det(tb[:3, :3])))
        rb = tb[:3, :3] / sb
        p_cam = ta[:3, :3] @ ca + ta[:3, 3]
        est.append(rb.T @ (p_cam - tb[:3, 3]))
    est = np.array(est)
    ga, gb = gt_poses(parquet, name_a), gt_poses(parquet, name_b)
    ca_gt = mesh_centre(SCANS / name_a / f"{name_a}.glb")
    gt = np.array([np.linalg.inv(gb[i]) @ np.r_[ga[i][:3, :3] @ ca_gt + ga[i][:3, 3], 1] for i in common])[:, :3]
    print(f"{len(common)} frames compared ({name_a} centre in {name_b} frame); gt travel "
          f"{np.linalg.norm(gt - gt[0], axis=1).max()*100:.1f} cm")
    for label, ws in (("rigid fit", False), ("Sim(3) fit", True)):
        s, Rm, t = umeyama(est, gt, with_scale=ws)
        err = np.linalg.norm((s * (Rm @ est.T)).T + t - gt, axis=1) * 100
        print(f"  {label:10s}: mean {err.mean():5.2f}  median {np.median(err):5.2f}  p90 {np.percentile(err, 90):5.2f}  "
              f"max {err.max():5.2f} cm" + (f"   (scale {s:.3f})" if ws else ""))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "mesh":
        cmd_mesh(*sys.argv[2:5])
    elif mode == "traj":
        cmd_traj(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
