"""Score a FlashCHORD rollout of a PUBLIC Track 3 episode against its mocap with the kit's OWN metric code
(metric_code/track_3/{AUC,SP-SR,MP-SR,RPE,MPPE}.py, the registered-mesh Track 3 path), i.e. a local stand-in for Kaggle.

  submission: the official evaluator's trajectory parquet (world 0, one step per frame) + our meshes, encoded exactly like
              tools/pack_submission.py (v2dlb.mesh_budget.budget_mesh -> 4096 vertices / faces, base64 npz).
  solution:   the episode's mocap poses (dataset [x,y,z,qw,qx,qy,qz] -> xyzw) + a reference-geometry payload built from the
              PUBLIC scans. The host payload is not released, so its fields are approximations:
                anchor  = scan surface centroid (the scorer's _surface_moments centre)
                points  = 2000 surface samples (ADD-AUC vertices), relative to the anchor in the scorer
                axis/order = per-object symmetry: --axial NAME (continuous spin about the scan's distinct principal axis,
                             e.g. cups), --order2 NAME (180-degree, e.g. the lid); default none.
Object slots follow the order given by --objects (use the evaluation roster order; for public episodes sorted names).

Usage (kit env):
  python score_public_episode.py KIT EPISODE ROLLOUT_PARQUET --objects blue_cup,plastic_dish_rack \
      --mesh blue_cup=PATH --mesh plastic_dish_rack=PATH [--axial blue_cup] [--order2 white_pot_lid]
"""
import argparse
import base64
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import trimesh

DATA = Path.home() / "v2d/v2d_data/track_3/public"


def encode(**arrays):
    s = io.BytesIO()
    np.savez_compressed(s, **arrays)
    return base64.b64encode(s.getvalue()).decode("ascii")


def symmetry_axis(m):
    """Axis of a body of revolution: the principal axis whose inertia eigenvalue differs from the other two."""
    w, V = np.linalg.eigh(m.moment_inertia)
    gaps = [abs(w[0] - w[1]) + abs(w[0] - w[2]), abs(w[1] - w[0]) + abs(w[1] - w[2]), abs(w[2] - w[0]) + abs(w[2] - w[1])]
    return V[:, int(np.argmax(gaps))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("episode", type=int)
    ap.add_argument("rollout")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--mesh", action="append", default=[])
    ap.add_argument("--axial", default="")
    ap.add_argument("--order2", default="")
    a = ap.parse_args()
    kit = Path(a.kit)
    sys.path.insert(0, str(kit))
    from v2dlb.mesh_budget import budget_mesh

    names = a.objects.split(",")
    meshes = dict(m.split("=", 1) for m in a.mesh)
    axial, order2 = set(filter(None, a.axial.split(","))), set(filter(None, a.order2.split(",")))
    ep = a.episode

    # --- solution: mocap ---
    rows = pq.read_table(DATA / "data/chunk-000" / f"episode_{ep:06d}.parquet",
                         columns=["observation.objects"]).column(0).to_pylist()
    sol = []
    for f, objs in enumerate(rows):
        for slot, n in enumerate(names):
            o = next(o for o in objs if o["name"] == n)
            x, y, z, qw, qx, qy, qz = o["pose"]
            sol.append({"row_id": f"track_3/e{ep:06d}/w000/f{f:06d}/o{slot}", "episode_index": ep, "world_index": 0,
                        "frame_index": f, "object_slot": slot, "object_name": n,
                        "ref_pos_x": x, "ref_pos_y": y, "ref_pos_z": z,
                        "ref_quat_x": qx, "ref_quat_y": qy, "ref_quat_z": qz, "ref_quat_w": qw})
    sol = pd.DataFrame(sol)
    geo = {"names": np.array(names)}
    for i, n in enumerate(names):
        mp = DATA / "mesh" / n / f"{n}.glb"
        mp = mp if mp.exists() else DATA / "mesh" / n / f"{n}_visual.glb"  # sweeper / wooden pieces
        m = trimesh.load(str(mp), force="mesh")
        if len(m.faces) > 20000:  # the scorer caps the decoded host payload at 16 MB (the real host's scans are reduced too)
            m = m.simplify_quadric_decimation(face_count=20000)
        geo[f"v{i}"] = np.asarray(m.vertices, float)
        geo[f"f{i}"] = np.asarray(m.faces, np.int64)
        anchor = np.asarray(m.center_mass if m.is_watertight else m.centroid, float)
        geo[f"anchor{i}"] = anchor
        geo[f"points{i}"] = np.asarray(trimesh.sample.sample_surface(m, 2000, seed=0)[0], float)
        axis = symmetry_axis(m) if (n in axial or n in order2) else np.zeros(3)
        geo[f"axis{i}"] = axis
        geo[f"order{i}"] = np.array(2 if n in order2 else (360 if n in axial else 1))  # scorer: 360 steps = continuous spin
    sol["reference_geometry_b64"] = None
    sol.loc[0, "reference_geometry_b64"] = encode(**geo)

    # --- submission: rollout + meshes ---
    t = pq.read_table(a.rollout).to_pandas()
    t = t[(t["environment_id"] == 0)]
    sub = []
    for slot, n in enumerate(names):
        b = t[t["object_body_name"] == n].sort_values("reference_frame")
        v, fcs = budget_mesh(meshes[n], faces=4096, vertices=4096)
        enc = encode(vertices=np.asarray(v, float), faces=np.asarray(fcs, float))
        for k, (_, r) in enumerate(b.iterrows()):
            sub.append({"row_id": f"track_3/e{ep:06d}/w000/f{int(r['reference_frame']):06d}/o{slot}",
                        "pos_x": r["achieved_position_x"], "pos_y": r["achieved_position_y"],
                        "pos_z": r["achieved_position_z"], "quat_x": r["achieved_quaternion_x"],
                        "quat_y": r["achieved_quaternion_y"], "quat_z": r["achieved_quaternion_z"],
                        "quat_w": r["achieved_quaternion_w"], "object_mesh_b64": enc if k == 0 else None})
    sub = pd.DataFrame(sub)
    for c, v in {"eval_script_sha256": "local", "eval_code_sha256": "local", "checkpoint_sha256": "local",
                 "packer_script_sha256": "local", "uploader_script_sha256": "local",
                 "code_commit_url": "https://github.com/local/local/commit/0000000"}.items():
        sub[c] = v
    missing = set(sol.row_id) - set(sub.row_id)
    print(f"episode {ep}: {len(sol)} solution rows, {len(sub)} submission rows, {len(missing)} missing")

    for metric in ("AUC", "SP-SR", "MP-SR", "RPE", "MPPE"):
        spec = importlib.util.spec_from_file_location(metric.replace("-", "_"), kit / "metric_code/track_3" / f"{metric}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  {metric:6s} {mod.score(sol.copy(), sub.copy(), 'row_id'):.4f}")


if __name__ == "__main__":
    main()
