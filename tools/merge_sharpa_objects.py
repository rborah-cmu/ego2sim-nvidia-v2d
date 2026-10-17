"""Merge single-object ManoSharpaData partitions of one clip into ONE multi-object partition for FlashCHORD.

The ego_recon loader/retargeter handle one object per clip, and the loader normalizes each clip's world separately
(xy centring + object bottom -> ground_z_offset, a pure translation when run with --no_ground_align). FlashCHORD supports
several independent rigid bodies in one reference (per-body URDF, all-zero articulation placeholder).

  * The PRIMARY sequence (first, normally the manipulated object) supplies the hands (MANO + retargeted robot fields).
  * Every other object's poses are moved into the primary's world: offset_s = t_primary - t_s, where
    t_s = sim position - bundle position at frame 0 (the loader's translation; rotations are checked to agree).
  * Contacts, per frame / side / hand link: the primary's contact if it has one (body 1), else the first other object in
    contact (body i, 1-indexed; positions shifted), else none (0, primary's arrays).
  * Support: the primary's table disk (tools/make_table_support.py) re-centred and enlarged to cover every body's travel;
    other primaries' higher disks are kept.

Usage (rg_retarget env):
  python merge_sharpa_objects.py OUT_PROCESSED_ROOT SEQ_ID STAGE_DIR \
      PRIMARY_SEQ::PARQUET::BUNDLE_RESULT_DIR [OTHER_SEQ::PARQUET::BUNDLE_RESULT_DIR ...]
Writes OUT_PROCESSED_ROOT/sequence_id=SEQ_ID/robot_name=sharpa_wave/SEQ_ID-0.parquet and
STAGE_DIR/SEQ_ID_sharpa_wave_support.usda.
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

SIDES = ("left", "right")
CONTACT = ("link_contact_positions", "link_contact_normals", "object_contact_positions", "object_contact_normals")
SHIFTED = ("link_contact_positions", "object_contact_positions")


def load(spec):
    seq, parquet, bundle = spec.split("::", 2)
    t = pq.read_table(parquet)
    row = t.to_pylist()[0]
    z = np.load(Path(bundle) / "result.npz")
    Tb = z["object_to_world_transform"][0].astype(float)
    p_sim = np.array(row["object_body_position"], float)[0, 0]
    q_sim = np.array(row["object_body_wxyz"], float)[0, 0]
    R_sim = R.from_quat([q_sim[1], q_sim[2], q_sim[3], q_sim[0]]).as_matrix()
    rot_gap = np.degrees(R.from_matrix(R_sim @ Tb[:3, :3].T).magnitude())
    return {"seq": seq, "parquet": parquet, "table": t, "row": row, "t": p_sim - Tb[:3, 3], "rot_gap": rot_gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("seq_id")
    ap.add_argument("stage_dir")
    ap.add_argument("seqs", nargs="+")
    ap.add_argument("--margin", type=float, default=0.25)
    a = ap.parse_args()
    S = [load(s) for s in a.seqs]
    P = S[0]
    for s in S:
        print(f"{s['seq']}: loader translation {np.round(s['t'], 4)}, loader-vs-bundle rotation gap {s['rot_gap']:.3f} deg")
        assert s["rot_gap"] < 0.5, "loader rotated the world (run it with --no_ground_align)"
    row = dict(P["row"])
    T = len(row["object_body_position"])
    pos, quat, names, safe, meshes, urdfs, radii = [], [], [], [], [], [], []
    for s in S:
        r, off = s["row"], P["t"] - s["t"]
        pos.append(np.array(r["object_body_position"], float) + off)
        quat.append(np.array(r["object_body_wxyz"], float))
        names += list(r["object_body_names"]); safe += list(r["safe_object_body_names"])
        meshes += list(r["object_mesh_paths"]); urdfs += list(r["object_urdf_paths"]); radii += list(r["object_mesh_radius"])
        s["off"] = off
    row["object_body_position"] = np.concatenate(pos, 1).tolist()
    row["object_body_wxyz"] = np.concatenate(quat, 1).tolist()
    row["object_body_names"], row["safe_object_body_names"] = names, safe
    row["object_mesh_paths"], row["object_urdf_paths"], row["object_mesh_radius"] = meshes, urdfs, radii
    row["object_name"] = row["safe_object_name"] = a.seq_id
    row["object_articulation"] = [0.0] * T

    for side in SIDES:
        pid = np.array(P["row"][f"mano_{side}_object_contact_part_ids"], float)
        out = {k: np.array(P["row"][f"mano_{side}_{k}"], float) for k in CONTACT}
        merged_pid = pid.copy()
        for i, s in enumerate(S[1:], start=2):
            spid = np.array(s["row"][f"mano_{side}_object_contact_part_ids"], float)
            take = (merged_pid == 0) & (spid > 0)
            merged_pid[take] = i
            for k in CONTACT:
                v = np.array(s["row"][f"mano_{side}_{k}"], float)
                if k in SHIFTED:
                    v = v + s["off"]
                out[k][take] = v[take]
        row[f"mano_{side}_object_contact_part_ids"] = merged_pid.astype(
            np.asarray(P["row"][f"mano_{side}_object_contact_part_ids"]).dtype).tolist()
        for k in CONTACT:
            row[f"mano_{side}_{k}"] = out[k].tolist()
        print(f"{side} hand contacts per body (link-frames): " +
              ", ".join(f"{n}={int((merged_pid == i + 1).sum())}" for i, n in enumerate(names)))

    d = Path(a.out_root) / f"sequence_id={a.seq_id}" / "robot_name=sharpa_wave"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.parquet"):
        f.unlink()
    out_t = pa.Table.from_pylist([row], schema=P["table"].schema)
    pq.write_table(out_t, d / f"{a.seq_id}-0.parquet")
    print(f"merged {len(names)} bodies {names} -> {d}")

    # support: the primary's table disk, re-centred/enlarged over every body's travel
    src = Path(a.stage_dir) / f"{P['seq']}_sharpa_wave_support.usda"
    st_in = Usd.Stage.Open(str(src))
    table = st_in.GetDefaultPrim().GetChild("table")
    h = table.GetAttribute("height").Get()
    top = table.GetAttribute("xformOp:translate").Get()[2] + h / 2
    allxy = np.concatenate([p[:, :, :2].reshape(-1, 2) for p in pos])
    c = (allxy.min(0) + allxy.max(0)) / 2
    radius = np.linalg.norm(allxy - c, axis=1).max() + a.margin
    out_usda = Path(a.stage_dir) / f"{a.seq_id}_sharpa_wave_support.usda"
    st = Usd.Stage.CreateNew(str(out_usda))
    root = UsdGeom.Xform.Define(st, "/support_surfaces")
    st.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    cyl = UsdGeom.Cylinder.Define(st, "/support_surfaces/table")
    cyl.CreateAxisAttr("Z"); cyl.CreateHeightAttr(h); cyl.CreateRadiusAttr(float(radius))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.35, 0.2)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), float(top - h / 2)))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    st.GetRootLayer().Save()
    print(f"support: table top {top:.3f} m, radius {radius:.2f} m -> {out_usda}")


if __name__ == "__main__":
    main()
