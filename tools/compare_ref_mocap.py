"""Compare reference (achieved:=reference rollout) object motion with mocap per object: when each starts moving, how far,
inter-object distance at frame 0. compare_ref_mocap.py EP ROLL.parquet"""
import sys, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
ep, roll = int(sys.argv[1]), sys.argv[2]
D = Path.home() / "v2d/v2d_data/track_3/public"
f = [p for p in glob.glob(str(D / "data/*/*.parquet")) if p.endswith(f"episode_{ep:06d}.parquet")][0]
m = pq.read_table(f).to_pandas()
objs = {}
for row in m["observation.objects"]:
    for o in row:
        objs.setdefault(o["name"], []).append(np.asarray(o["pose"][:3], float) if o["visible"] else np.full(3, np.nan))
r = pq.read_table(roll).to_pandas()
ours = {n: g.sort_values("step")[["achieved_position_x", "achieved_position_y", "achieved_position_z"]].to_numpy()
        for n, g in r.groupby("object_body_name", observed=True)}
def summ(tag, P):
    P = np.asarray(P); d = np.linalg.norm(P - P[0], axis=1) * 100
    mv = np.nonzero(d > 2)[0]
    print(f"  {tag:8s} start move f{mv[0] if len(mv) else '-'}  max disp {np.nanmax(d):5.1f} cm  dz max {100*np.nanmax(P[:,2]-P[0,2]):5.1f}  "
          + " ".join(f"{x:5.1f}" for x in d[::25]))
for n in sorted(objs):
    print(n); summ("mocap", objs[n])
    if n in ours: summ("ours", ours[n])
names = sorted(objs)
if len(names) == 2:
    a, b = names
    print("frame-0 inter-object distance mocap %.1f cm, ours %.1f cm" % (100*np.linalg.norm(objs[a][0]-objs[b][0]), 100*np.linalg.norm(ours[a][0]-ours[b][0])))