"""Local CD-O check with the submission kit's own code: pack a candidate mesh exactly as
tools/pack_submission.py does (v2dlb.mesh_budget.budget_mesh, 4096 faces / 4096 vertices) and score it
with the official metric function (metric_code/track_3/CD-O.py::_object_chamfer_cm) against a reference
mesh, using the per-(episode, slot) seeds Kaggle uses for every eval episode that contains the object.

Usage: python local_cd_o.py KIT_ROOT OBJECT_NAME CANDIDATE_MESH REFERENCE_MESH [--label NAME] [--cylinder R H]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("object")
    ap.add_argument("candidate")
    ap.add_argument("reference")
    a = ap.parse_args()
    kit = Path(a.kit)
    sys.path.insert(0, str(kit))
    from v2dlb.mesh_budget import budget_mesh  # the packer's decimation + padding
    cdo = load_module(kit / "metric_code/track_3/CD-O.py", "cdo_metric")  # the official scorer

    roster = json.load(open(kit / "data/track_3_evaluation_objects.json"))["episodes"]
    eps = [(int(e), objs.index(a.object)) for e, objs in roster.items() if a.object in objs]

    def packed(path):
        v, f = budget_mesh(path, faces=4096, vertices=4096)
        return v, f

    def raw(path):
        import trimesh
        m = trimesh.load(path, force="mesh", process=False)
        return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)

    cand = packed(a.candidate)
    refs = {"reference as-is": raw(a.reference), "reference packed to 4096": packed(a.reference)}
    real_faces = int((np.linalg.norm(np.cross(cand[0][cand[1][:, 1]] - cand[0][cand[1][:, 0]],
                                              cand[0][cand[1][:, 2]] - cand[0][cand[1][:, 0]]), axis=1) > 0).sum())
    print(f"{a.object}: eval episodes/slots {eps}; candidate packed to {len(cand[0])} verts / {len(cand[1])} faces "
          f"({real_faces} non-padding)")
    for label, ref in refs.items():
        vals = [cdo._object_chamfer_cm(ref, cand, cdo._object_seed(e, s)) for e, s in eps]
        print(f"  CD-O vs {label:26s}: mean {np.mean(vals):.3f} cm  per episode "
              + ", ".join(f"e{e}={v:.3f}" for (e, _), v in zip(eps, vals)))


if __name__ == "__main__":
    main()
