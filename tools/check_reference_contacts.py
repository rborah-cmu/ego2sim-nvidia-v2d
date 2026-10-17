"""Hand-object geometry check of a retargeted ManoSharpaData reference: per frame and hand, the fingertip gap to each object
(MANO tips and the robot's *_tip_site frames; negative = penetration) and wrist jumps between frames.

Catches the two failure modes found on ep12: fingertips 2-3 cm inside the object (the sim fingers can't follow) and
wrist teleports from failed hand depth alignment (single-frame jumps of tens of cm).

Usage (rg_retarget env): python check_reference_contacts.py PARTITION_DIR [--every 10] [--jump 0.05]
"""
import argparse
import glob

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

TIPS_MANO = [4, 8, 12, 16, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("partition")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--jump", type=float, default=0.05)
    a = ap.parse_args()
    r = pq.read_table(glob.glob(a.partition + "/*.parquet")[0]).to_pylist()[0]
    pos = np.array(r["object_body_position"])
    quat = np.array(r["object_body_wxyz"])
    T, B = pos.shape[:2]
    pqs = [trimesh.proximity.ProximityQuery(trimesh.load(p, force="mesh")) for p in r["object_mesh_paths"]]
    names = r["object_body_names"]
    out = {}
    for s in ("left", "right"):
        J = np.array(r[f"mano_{s}_joints"])[:, TIPS_MANO]
        fn = r[f"{s}_robot_frame_names"]
        RF = np.array(r[f"robot_{s}_frames"])[:, [i for i, n in enumerate(fn) if n.endswith("tip_site")], :3]
        g = np.full((T, B, 2), np.inf)
        for t in range(T):
            for b in range(B):
                Rm = R.from_quat([quat[t, b, 1], quat[t, b, 2], quat[t, b, 3], quat[t, b, 0]]).as_matrix()
                for k, P in enumerate((J[t], RF[t])):
                    g[t, b, k] = -pqs[b].signed_distance((P - pos[t, b]) @ Rm).max()
        w = np.array(r[f"robot_{s}_wrist_position"])
        jumps = np.linalg.norm(np.diff(w, axis=0), axis=1)
        out[s] = (g, jumps)
        bad = np.nonzero(jumps > a.jump)[0]
        print(f"{s}: wrist jumps > {a.jump*100:.0f} cm/frame at frames {bad.tolist()[:20]} (max {jumps.max()*100:.1f} cm)")
    print("frame | " + " | ".join(f"{s[0]} {n}: mano/robot tip gap cm" for s in ("left", "right") for n in names))
    for t in range(0, T, a.every):
        cells = [f"{out[s][0][t, b, 0]*100:6.1f}/{out[s][0][t, b, 1]*100:6.1f}" for s in ("left", "right") for b in range(B)]
        print(f"{t:4d} | " + " | ".join(cells))
    for s in ("left", "right"):
        g = out[s][0]
        for b, n in enumerate(names):
            close = g[:, b, 0] < 0.03
            if close.any():
                print(f"{s} on {n}: {close.sum()} frames within 3 cm; MANO penetration in those: median "
                      f"{np.median(np.minimum(g[close, b, 0], 0))*100:.1f} cm, worst {g[close, b, 0].min()*100:.1f} cm; "
                      f"robot worst {g[close, b, 1].min()*100:.1f} cm")


if __name__ == "__main__":
    main()
