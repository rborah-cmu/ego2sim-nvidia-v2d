"""Fix a static object's FoundationPose depth error (placed too far along the camera ray -> sunk into the table).
Table plane from a REFERENCE object resting on it (its mesh up axis = smallest extent; plane through its lowest point at
frame 0). Each frame's camera-frame translation t is scaled to s*t (same image position) with one s chosen so the target's
lowest point at frame 0 lies on that plane. Writes the corrected poses to OUT_DIR.
  fix_static_depth.py EPDIR TARGET_OBJ REFERENCE_OBJ MESH_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
E, tgt, ref, M, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
c2w = np.load(E / "camera_vo.npz")["camera_to_world"]
def load(o, i):
    j = json.load(open(E / f"fp_{o}/{i:06d}.json")); w, x, y, z = j["rotation"]
    T = np.eye(4); T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix(); T[:3, 3] = j["translation"]; return T, j
def verts(o):
    m = trimesh.load(str(M / f"{o}.obj"), force="mesh"); return np.asarray(m.vertices), int(np.argmin(m.extents))
vr, ar = verts(ref); vt, _ = verts(tgt)
Tr, _ = load(ref, 0); W = c2w[0] @ Tr
n = W[:3, ar] / np.linalg.norm(W[:3, ar])
pts = vr @ W[:3, :3].T + W[:3, 3]
if np.median(pts @ n) < W[:3, 3] @ n:  # orient n away from the table (towards the object's bulk is ambiguous: use camera)
    pass
cam = c2w[0][:3, 3]
if (cam - W[:3, 3]) @ n < 0:
    n = -n  # normal points to the camera side (up)
h0 = (pts @ n).min()  # table height along n
T0, _ = load(tgt, 0)
def bottom(s):
    Ti = T0.copy(); Ti[:3, 3] *= s; Wt = c2w[0] @ Ti; return (vt @ Wt[:3, :3].T + Wt[:3, 3]) @ n - h0
lo, hi = 0.5, 1.5
for _ in range(60):
    mid = (lo + hi) / 2
    # bottom height decreases as s grows (farther from camera -> lower) for a downward-looking camera
    if bottom(mid).min() > 0: lo = mid
    else: hi = mid
s = (lo + hi) / 2
print(f"{tgt}: bottom before {100 * bottom(1.0).min():.1f} cm, scale {s:.4f} -> {100 * bottom(s).min():.1f} cm "
      f"(depth {100 * T0[2, 3]:.1f} -> {100 * s * T0[2, 3]:.1f} cm)")
out.mkdir(parents=True, exist_ok=True)
for f in sorted((E / f"fp_{tgt}").glob("*.json")):
    j = json.load(open(f)); j["translation"] = [float(s * x) for x in j["translation"]]; json.dump(j, open(out / f.name, "w"), indent=1)