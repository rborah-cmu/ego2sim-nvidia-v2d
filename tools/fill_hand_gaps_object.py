"""Fill long hand-track gaps where the hand is holding an object, by carrying the hand rigidly with that object.

WiLoR often loses a hand while it grips an object (occluded / foreshortened: ep41 left hand frames 55-115 and 160-180
while lifting the pot), longer than the interpolation limit, so the reference has a one-handed lift the policy cannot
reproduce. For each gap (a, b) between records, if the wrist is within --near of an object's origin at BOTH ends, the
hand pose is expressed in that object's camera-frame pose (FoundationPose track) at a and b, interpolated there
(lerp / slerp), and mapped back with the object's pose at every gap frame. Other gaps up to --max-free frames are
interpolated in world coordinates (camera_to_world). Longer free gaps stay missing. With FP_DIR=MESH the distance is to
the nearest mesh vertex (large objects held by a handle sit far from their origin), else to the origin.

MANO convention (manotorch, center_idx=None, as in v2d_hamer align_hands): verts = R (v - j0) + j0 + cam_t, global
rotation about the rest-pose root joint j0; left hands are the right model mirrored in x, so their effective rotation is
M R M and pivot M j0 (M = diag(-1, 1, 1)). Rigid hand transform T = (R_eff, cam_t + p - R_eff p).

  fill_hand_gaps_object.py CAM_NPZ IN_DIR OUT_DIR --objects FP_DIR[=MESH][,FP_DIR[=MESH]] --frames N [--near 0.25] [--max-free 30]
Run in the v2d_hamer env (manotorch).
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

MANO = str(Path.home() / "v2d/mano")
M = np.diag([-1.0, 1.0, 1.0])


def root_joint(mano, betas):
    out = mano(torch.zeros(1, 48), torch.tensor(betas, dtype=torch.float32)[None])
    return out.joints[0, 0].detach().numpy().astype(float)


def rigid(rec, j0):
    Rr = R.from_rotvec(rec["mano"]["global_orient"]).as_matrix()
    if rec["is_right"]:
        Re, p = Rr, j0
    else:
        Re, p = M @ Rr @ M, M @ j0
    T = np.eye(4)
    T[:3, :3] = Re
    T[:3, 3] = np.asarray(rec["camera"]["cam_t"], float) + p - Re @ p
    return T, p


def unrigid(T, p, is_right):
    Re = T[:3, :3]
    Rr = Re if is_right else M @ Re @ M
    return R.from_matrix(Rr).as_rotvec().tolist(), (T[:3, 3] - p + Re @ p).tolist()


def fp_pose(d, i):
    f = Path(d) / f"{i:06d}.json"
    if not f.exists():
        return None
    j = json.load(open(f))
    w, x, y, z = j["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = j["translation"]
    return T


def interp_T(Ta, Tb, s):
    rot = Slerp([0, 1], R.from_matrix([Ta[:3, :3], Tb[:3, :3]]))([s])[0].as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = (1 - s) * Ta[:3, 3] + s * Tb[:3, 3]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam_npz")
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--near", type=float, default=0.12)
    ap.add_argument("--max-free", type=int, default=30)
    ap.add_argument("--blend", type=int, default=10)
    a = ap.parse_args()
    from manotorch.manolayer import ManoLayer
    mano = ManoLayer(rot_mode="axisang", use_pca=False, side="right", center_idx=None, mano_assets_root=MANO)
    c2w = np.load(a.cam_npz)["camera_to_world"]
    objs, verts = [], {}
    for spec in [s for s in a.objects.split(",") if s]:
        d, _, mesh = spec.partition("=")
        objs.append(d)
        if mesh:
            import trimesh
            m = trimesh.load(mesh, force="mesh")
            v = np.asarray(m.vertices, float)
            verts[d] = v[np.random.default_rng(0).choice(len(v), min(len(v), 4000), replace=False)]

    def dist(d, O, w):
        if d in verts:
            return np.linalg.norm(verts[d] @ O[:3, :3].T + O[:3, 3] - w, axis=1).min()
        return np.linalg.norm(O[:3, 3] - w)
    recs = {int(f.stem): json.load(open(f)) for f in sorted(Path(a.in_dir).glob("*.json"))}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, r in recs.items():
        json.dump(r, open(out / f"{i:06d}.json", "w"))
    have = sorted(recs)
    log = []
    for a_i, b_i in zip(have[:-1], have[1:]):
        if b_i - a_i <= 1:
            continue
        ra, rb = recs[a_i], recs[b_i]
        j0 = root_joint(mano, ra["mano"]["betas"])
        Ta, p = rigid(ra, j0)
        Tb, _ = rigid(rb, j0)
        wa, wb = Ta[:3, :3] @ p + Ta[:3, 3], Tb[:3, :3] @ p + Tb[:3, 3]  # wrist (root joint) in camera coords
        best = None
        for d in objs:
            Oa, Ob = fp_pose(d, a_i), fp_pose(d, b_i)
            if Oa is None or Ob is None:
                continue
            da, db = dist(d, Oa, wa), dist(d, Ob, wb)
            # both ends near: carry, interpolating in the object frame; one end near: hold that end's grasp (side "a"
            # or "b") and blend in world to the other end's pose over the last/first --blend frames
            if max(da, db) < a.near:
                cand = (max(da, db), d, Oa, Ob, "ab")
            elif min(da, db) < a.near:
                cand = (min(da, db) + 1.0, d, Oa, Ob, "a" if da < db else "b")
            else:
                continue
            if best is None or cand[0] < best[0]:
                best = cand
        fa = np.asarray(ra["mano"]["hand_pose"], float).reshape(-1)
        fb = np.asarray(rb["mano"]["hand_pose"], float).reshape(-1)
        n_gap = b_i - a_i - 1
        if best is not None:
            _, d, Oa, Ob, ends = best
            rel_a, rel_b = np.linalg.inv(Oa) @ Ta, np.linalg.inv(Ob) @ Tb
            mode = f"object {Path(d).name} (ends {ends}, wrist {100 * (best[0] % 1.0):.0f} cm)"
        elif n_gap <= a.max_free:
            Wa, Wb = c2w[a_i] @ Ta, c2w[b_i] @ Tb
            mode = "world"
        else:
            log.append(f"{a_i + 1}-{b_i - 1} left missing ({n_gap} frames, no object within {a.near} m)")
            continue
        filled = 0
        for i in range(a_i + 1, b_i):
            s = (i - a_i) / (b_i - a_i)
            if best is not None:
                Oi = fp_pose(d, i)
                if Oi is None:
                    continue
                if ends == "ab":
                    Ti = Oi @ interp_T(rel_a, rel_b, s)
                else:
                    Ti = Oi @ (rel_a if ends == "a" else rel_b)
                    k = (b_i - i) if ends == "a" else (i - a_i)  # frames to the far end
                    if k <= a.blend:
                        far = Tb if ends == "a" else Ta
                        far_i, far_f = (b_i, far) if ends == "a" else (a_i, far)
                        Wc, Wf = c2w[i] @ Ti, c2w[far_i] @ far_f
                        Ti = np.linalg.inv(c2w[i]) @ interp_T(Wf, Wc, (k - 1) / a.blend)
            else:
                Ti = np.linalg.inv(c2w[i]) @ interp_T(Wa, Wb, s)
            rec = copy.deepcopy(ra)
            rec["frame_idx"] = i
            rec["mano"]["global_orient"], rec["camera"]["cam_t"] = unrigid(Ti, p, ra["is_right"])
            rec["mano"]["hand_pose"] = ((1 - s) * fa + s * fb).reshape(-1, 3).tolist()
            rec["interpolated"] = True
            rec["object_carried"] = best is not None
            json.dump(rec, open(out / f"{i:06d}.json", "w"))
            filled += 1
        log.append(f"{a_i + 1}-{b_i - 1} ({n_gap}) filled {filled} via {mode}")
    n_out = len(list(out.glob("*.json")))
    print(f"{a.in_dir}: {len(recs)} -> {n_out}/{a.frames} frames; " + "; ".join(log))


if __name__ == "__main__":
    main()
