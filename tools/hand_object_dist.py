"""Per object & frame: FP displacement from frame 0 and min wrist->surface distance (objfill hands). Calibrates the
'objects only move when touched' prior.  hand_object_dist.py EP OBJ1,OBJ2"""
import sys, json
import numpy as np, torch, trimesh
from pathlib import Path
sys.path.insert(0, str(Path.home() / "v2d/tools"))
from fill_hand_gaps_object import rigid, root_joint, fp_pose, MANO
from manotorch.manolayer import ManoLayer
ep, objs = sys.argv[1], sys.argv[2].split(",")
E = Path.home() / f"v2d/ego2sim/ep{ep}"; SC = Path.home() / "v2d/v2d_data/track_3/public/mesh"
mano = ManoLayer(rot_mode="axisang", use_pca=False, side="right", center_idx=None, mano_assets_root=MANO)
n = len(list((E / "hands/frames").glob("*.jpg")))
W = {}
for s in ("left", "right"):
    W[s] = np.full((n, 3), np.nan); j0 = None
    for f in sorted((E / f"hands/objfill/{s}").glob("*.json")):
        r = json.load(open(f)); j0 = root_joint(mano, r["mano"]["betas"]) if j0 is None else j0
        T, p = rigid(r, j0); W[s][int(f.stem)] = T[:3, :3] @ p + T[:3, 3]
for o in objs:
    mp = SC / o / f"{o}.glb"; mp = mp if mp.exists() else SC / o / f"{o}_visual.glb"
    v = np.asarray(trimesh.load(str(mp), force="mesh").vertices); v = v[np.random.default_rng(0).choice(len(v), min(4000, len(v)), replace=False)]
    O0 = fp_pose(E / f"fp_{o}", 0); rows = []
    for i in range(n):
        O = fp_pose(E / f"fp_{o}", i)
        # displacement in camera frame is confounded by head motion -> use world via camera_to_world
        c2w = np.load(E / "camera_vo.npz")["camera_to_world"] if i == 0 and not rows else c2w
        ow = c2w[i] @ O; ow0 = c2w[0] @ O0
        disp = np.linalg.norm(ow[:3, 3] - ow0[:3, 3]) * 100
        pts = v @ O[:3, :3].T + O[:3, 3]
        d = min(np.linalg.norm(pts - W[s][i], axis=1).min() if not np.isnan(W[s][i]).any() else 9 for s in W) * 100
        rows.append((disp, d))
    rows = np.array(rows)
    print(o, "disp/handdist every 10:", " ".join(f"{a:.0f}/{b:.0f}" for a, b in rows[::10]))