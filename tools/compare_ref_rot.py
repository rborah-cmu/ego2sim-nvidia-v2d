import sys, glob, json
import numpy as np, pyarrow.parquet as pq, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
ep, roll = int(sys.argv[1]), sys.argv[2]
D = Path.home() / "v2d/v2d_data/track_3/public"
f = [p for p in glob.glob(str(D / "data/*/*.parquet")) if p.endswith(f"episode_{ep:06d}.parquet")][0]
m = pq.read_table(f).to_pandas()
mq = {}
for row in m["observation.objects"]:
    for o in row:
        w, x, y, z = o["pose"][3:]; mq.setdefault(o["name"], []).append(R.from_quat([x, y, z, w]))
r = pq.read_table(roll).to_pandas()
oq = {n: [R.from_quat(q) for q in g.sort_values("step")[["achieved_quaternion_x","achieved_quaternion_y","achieved_quaternion_z","achieved_quaternion_w"]].to_numpy()] for n, g in r.groupby("object_body_name", observed=True)}
names = sorted(oq)
for n in names:
    mesh = trimesh.load(str(D / f"mesh/{n}/{n}_visual.glb"), force="mesh") if (D / f"mesh/{n}/{n}_visual.glb").exists() else trimesh.load(str(D / f"mesh/{n}/{n}.glb"), force="mesh")
    print(n, "extents cm", np.round(100 * mesh.extents, 1))
    # world-frame-independent: rotation of each object relative to its own frame 0, ours vs mocap
    errs = [np.degrees(((oq[n][i] * oq[n][0].inv()) * (mq[n][i] * mq[n][0].inv()).inv()).magnitude()) for i in range(0, len(oq[n]), 25)]
    print("  relative-to-frame-0 rotation disagreement (deg) every 25 frames:", np.round(errs, 0))
if len(names) == 2:
    a, b = names
    ro = oq[a][0].inv() * oq[b][0]; rm = mq[a][0].inv() * mq[b][0]
    print("frame-0 relative rotation a->b: ours vs mocap differ by %.0f deg" % np.degrees((ro * rm.inv()).magnitude()))