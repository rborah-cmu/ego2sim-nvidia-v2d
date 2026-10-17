"""Prepare and post-process AgentSTAR captures for a Track 3 episode (riddle side).

  inputs  — frames from the undistorted cam_a video + per-object SAM2 masks + occluder masks
            (hands, plus the other objects) in AgentSTAR's make_capture input layout.
  metric  — after `make_capture.py --depth`, rescale the capture to metres with our stereo depth:
            s = median over keyframes of (stereo Z / Pi3X Z) on pixels valid in both, then
            c2w translations *= s and depth pointmaps *= s (Pi3X geometry is up to scale).

Usage:
  python agentstar_prep.py inputs UNDIST_MP4 MASKS_DIR HAND_MASKS_DIR OUT_ROOT OBJ_ID:NAME [OBJ_ID:NAME ...]
  python agentstar_prep.py metric CAPTURE_DIR STEREO_DEPTH_DIR
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def cmd_inputs(video, masks_dir, hands_dir, out_root, objs):
    masks_dir, hands_dir, out_root = Path(masks_dir), Path(hands_dir), Path(out_root)
    objs = [o.split(":") for o in objs]
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    hand_ids = sorted(p.name for p in hands_dir.iterdir() if p.is_dir())
    for oid, name in objs:
        src = out_root / name
        for sub in ("frames", "mask_object", "mask_hand"):
            (src / sub).mkdir(parents=True, exist_ok=True)
        others = [o for o, _ in objs if o != oid]
        n_obj = 0
        for i, img in enumerate(frames):
            stem = f"{i:06d}"
            m = mask(masks_dir / oid / f"{stem}.png")
            if m is None or not m.any():
                continue  # make_capture drops frames without an object mask anyway
            cv2.imwrite(str(src / "frames" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(src / "mask_object" / f"{stem}.png"), m.astype(np.uint8) * 255)
            occ = np.zeros(m.shape, bool)
            for h in hand_ids:
                hm = mask(hands_dir / h / f"{stem}.png")
                if hm is not None:
                    occ |= hm
            for o in others:  # other tracked objects occlude this one
                om = mask(masks_dir / o / f"{stem}.png")
                if om is not None:
                    occ |= om
            occ &= ~m
            cv2.imwrite(str(src / "mask_hand" / f"{stem}.png"), occ.astype(np.uint8) * 255)
            n_obj += 1
        print(f"{name}: {n_obj}/{len(frames)} frames with object mask -> {src}")


def decode_v2d_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[raw >= 65535] = 0
    return d


def cmd_metric(capture, stereo_dir):
    capture, stereo_dir = Path(capture), Path(stereo_dir)
    man = json.load(open(capture / "capture.json"))
    assert man.get("depth") == "pi3x", "run make_capture.py with --depth"
    kfs = json.load(open(capture / "tracking" / "keyframes.json"))
    cams = dict(np.load(capture / "tracking" / "cameras.npz"))
    Wp, Hp = (int(v) for v in cams["size"])
    ratios = []
    for kf in kfs:
        stem = Path(kf["frame_name"]).stem
        z = np.load(capture / "depth" / f"{stem}.npz")
        zp, keep = z["local"][..., 2], z["keep"]
        st = decode_v2d_depth(stereo_dir / f"{stem}.png")
        st = cv2.resize(st, (Wp, Hp), interpolation=cv2.INTER_NEAREST)
        ok = keep & (st > 0.2) & (st < 3.0) & (zp > 1e-6)
        if ok.sum() > 500:
            ratios.append(np.median(st[ok] / zp[ok]))
    s = float(np.median(ratios))
    print(f"stereo/Pi3X scale: median {s:.4f}  p10 {np.percentile(ratios, 10):.4f}  p90 {np.percentile(ratios, 90):.4f} "
          f"over {len(ratios)} keyframes")
    if (capture / "tracking" / "cameras_pi3x_unscaled.npz").exists():
        raise SystemExit("capture already rescaled")
    np.savez(capture / "tracking" / "cameras_pi3x_unscaled.npz", **cams)
    cams["c2w"] = cams["c2w"].copy()
    cams["c2w"][:, :3, 3] *= s
    np.savez(capture / "tracking" / "cameras.npz", **cams)
    for kf in kfs:
        p = capture / "depth" / f"{Path(kf['frame_name']).stem}.npz"
        z = dict(np.load(p))
        z["local"] = (z["local"] * s).astype(np.float32)
        np.savez_compressed(p, **z)
    man.setdefault("source", {})["metric_scale_from_stereo"] = s
    json.dump(man, open(capture / "capture.json", "w"), indent=2)
    K = cams["intrinsics"][0]
    print(f"Pi3X fx on its {Wp}x{Hp} grid = {K[0, 0]:.1f} -> {K[0, 0] * man['image_size'][0] / Wp:.1f} at full res "
          f"(factory undistorted fx 1149.8)")


if __name__ == "__main__":
    if sys.argv[1] == "inputs":
        cmd_inputs(*sys.argv[2:6], sys.argv[6:])
    elif sys.argv[1] == "metric":
        cmd_metric(*sys.argv[2:4])
    else:
        raise SystemExit(__doc__)
