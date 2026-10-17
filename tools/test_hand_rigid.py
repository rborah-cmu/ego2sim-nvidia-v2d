import json, sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from fill_hand_gaps_object import rigid, root_joint, MANO
from manotorch.manolayer import ManoLayer
mano = ManoLayer(rot_mode="axisang", use_pca=False, side="right", center_idx=None, mano_assets_root=MANO)
for f in sys.argv[2:]:
    rec = json.load(open(f))
    pose = np.concatenate([rec["mano"]["global_orient"], np.asarray(rec["mano"]["hand_pose"]).reshape(-1)]).astype(np.float32)
    b = torch.tensor(rec["mano"]["betas"], dtype=torch.float32)[None]
    v = mano(torch.from_numpy(pose)[None], b).verts[0].numpy().astype(float)
    p0 = pose.copy(); p0[:3] = 0
    v0 = mano(torch.from_numpy(p0)[None], b).verts[0].numpy().astype(float)
    if not rec["is_right"]: v[:, 0] *= -1; v0[:, 0] *= -1
    v += np.asarray(rec["camera"]["cam_t"])
    T, p = rigid(rec, root_joint(mano, rec["mano"]["betas"]))
    pred = v0 @ T[:3, :3].T + T[:3, 3]
    print(f, "is_right", rec["is_right"], "max vertex err (mm)", 1000 * np.abs(pred - v).max())