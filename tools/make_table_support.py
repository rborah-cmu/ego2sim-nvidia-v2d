"""Add the TABLE (from the bundle's stereo table plane) to a FlashCHORD support stage.

support_recon only places disks where an object was seen resting, so an object that starts on the table but is not
detected as still there (or whose rest is short) has nothing under it in sim and falls to the floor. We know the table
plane: tools/build_ego_bundle.py gravity-aligned the world so the table is z = 0 (bundle world). The loader then moves the
scene rigidly (xy centring, object bottom at frame 0 -> --ground_z_offset), which we recover from the retargeted parquet.

  table_top_sim = minz_sim(object, frame 0) - minz_bundle(object, frame 0)

A Cylinder collision disk (top = table_top_sim) centred on the object's xy travel is added; existing support disks more than
2 cm above the table are kept (e.g. a rack the object rests in), lower ones are dropped (they duplicate the table).

Usage (rg_retarget env, needs pxr):
  python make_table_support.py BUNDLE_RESULT_DIR PROCESSED_PARQUET MESH_OBJ SUPPORT_USDA [--margin 0.25]
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_result_dir")
    ap.add_argument("parquet")
    ap.add_argument("mesh")
    ap.add_argument("usda")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--height", type=float, default=0.01)
    a = ap.parse_args()
    v = np.asarray(trimesh.load(a.mesh, force="mesh").vertices)
    z = np.load(Path(a.bundle_result_dir) / "result.npz")
    Tb = z["object_to_world_transform"][0].astype(float)
    minz_bundle = (v @ Tb[:3, :3].T + Tb[:3, 3])[:, 2].min()
    row = pq.read_table(a.parquet, columns=["object_body_position", "object_body_wxyz"]).to_pylist()[0]
    pos = np.array(row["object_body_position"], float)[:, 0]
    q = np.array(row["object_body_wxyz"], float)[:, 0]
    Rs0 = R.from_quat([q[0, 1], q[0, 2], q[0, 3], q[0, 0]]).as_matrix()
    minz_sim = (v @ Rs0.T + pos[0])[:, 2].min()
    # never above the object's lowest point at frame 0 (stereo/tracking noise can put it a few mm inside the table)
    top = min(minz_sim - minz_bundle, minz_sim)
    c = pos[:, :2].mean(0)
    radius = np.linalg.norm(pos[:, :2] - c, axis=1).max() + a.margin

    stage = Usd.Stage.Open(a.usda) if Path(a.usda).exists() else Usd.Stage.CreateNew(a.usda)
    if not stage.GetDefaultPrim():
        root = UsdGeom.Xform.Define(stage, "/support_surfaces")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.GetDefaultPrim()
    kept, dropped = [], []
    for p in list(root.GetChildren()):
        if p.GetName() == "table":
            stage.RemovePrim(p.GetPath())
            continue
        t = p.GetAttribute("xformOp:translate").Get()
        if t is not None and t[2] <= top + 0.02:
            dropped.append(p.GetName())
            stage.RemovePrim(p.GetPath())
        else:
            kept.append(p.GetName())
    cyl = UsdGeom.Cylinder.Define(stage, root.GetPath().AppendChild("table"))
    cyl.CreateAxisAttr("Z")
    cyl.CreateHeightAttr(a.height)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.35, 0.2)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), float(top - a.height / 2)))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    stage.GetRootLayer().Save()
    print(f"table top z={top:.3f} m (object bottom {minz_bundle*100:.1f} cm above the table at frame 0), "
          f"radius {radius:.2f} m at ({c[0]:.3f}, {c[1]:.3f}); kept {kept}, dropped {dropped} -> {a.usda}")


if __name__ == "__main__":
    main()
