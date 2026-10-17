"""AgentSTAR mesh -> metric mesh for FoundationPose / the simulator / the submission.
object.glb (canonical units) x pose.json "scale" -> metres; each part's PBR baseColorFactor baked into vertex colours
(v2d Mesh.load crashes on an OBJ material without colours). Writes OUT.obj (+ OUT.glb next to it with --glb).
  agentstar_mesh_export.py MESH_DIR(with object.glb + pose.json) OUT.obj [--glb]"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
src, out = Path(sys.argv[1]), Path(sys.argv[2])
scale = float(json.load(open(src / "pose.json"))["scale"])
scene = trimesh.load(src / "object.glb", force="scene", process=False)
parts = []
for name, geom in scene.geometry.items():
    for node in scene.graph.geometry_nodes.get(name, [None]):
        g = geom.copy()
        if node is not None:
            T, _ = scene.graph[node]; g.apply_transform(T)
        col = np.array([180, 180, 180, 255], np.uint8)
        mat = getattr(getattr(g, "visual", None), "material", None)
        bc = getattr(mat, "baseColorFactor", None) if mat is not None else None
        if bc is not None:
            bc = np.asarray(bc, float); col = (bc * (255 if bc.max() <= 1.0 else 1)).astype(np.uint8)
        g.visual = trimesh.visual.ColorVisuals(g, vertex_colors=np.tile(col, (len(g.vertices), 1)))
        parts.append(g)
m = trimesh.util.concatenate(parts)
m.apply_scale(scale)
out.parent.mkdir(parents=True, exist_ok=True)
m.export(out)
if "--glb" in sys.argv:
    m.export(out.with_suffix(".glb"))
print(f"{out}: {len(m.vertices)} verts {len(m.faces)} faces, scale {scale:.4f}, extents cm {np.round(m.extents * 100, 1)}, watertight {m.is_watertight}")