"""Hand reconstruction for one ego episode with the official V2D modules, fed by OUR inputs:
undistorted cam_a video + factory intrinsics + SGBM stereo depth (tools/ego_stereo_depth.py), and SAM2 hand masks
(one dir per hand track) + object masks. Mirrors steps 3, 9, 11, 13 of v2d_pipelines/run_ego_wilor.py:

  1. frames/                          extract the video's frames (the matcher reads them)
  2. wilor_raw/<frame>.json           WiLoR on every frame (bbox + MANO, virtual camera)
  3. hand_tracks.json                 our SAM2 hand tracks (id, role=hand, is_right)
  4. wilor/<track>/<frame>.json       WiLoR detections matched to SAM2 hand masks (silhouette IoU)
  5. wilor_aligned/<track>/…          metric depth alignment (real intrinsics + stereo depth; objects masked)
  6. wilor_aligned_filled/<track>/…   interpolation of short gaps (<= --max-gap frames)

Output dirs 6/<track> are what tools/build_ego_bundle.py --left-hand/--right-hand take.

Usage (recon_host env, from anywhere):
  python ego_hands.py STEREO_DIR OUT_DIR --hands 0:left,1:right   (or ids only, e.g. 0,1,2: side voted from WiLoR) --hand-masks DIR --object-masks DIR[,DIR] [--gpu 0]
  --hand-masks DIR holds <track_id>/<frame>.png (SAM2 output layout).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

RECON = Path.home() / "v2d/vtd_pinned/reconstruction"
MANO = str(Path.home() / "v2d/mano")
WILOR_W = str(RECON / "data/weights/wilor")


def step(label, done):
    print(f"[{'skip' if done else 'run '}] {label}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stereo_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--hands", required=True, help="track_id:left|right,... e.g. 0:left,1:right")
    ap.add_argument("--hand-masks", required=True)
    ap.add_argument("--object-masks", default="")
    ap.add_argument("--min-iou", type=float, default=0.1)
    ap.add_argument("--mask-min-pixels", type=int, default=256)
    ap.add_argument("--max-gap", type=int, default=15)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--depth-dir", default="", help="depth PNGs for alignment (default STEREO_DIR/stereo_depth; use a "
                    "fused stereo+MoGe dir from tools/fuse_depth.py so hands outside the stereo overlap get depth)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from v2d.hamer.docker.run_align_hands import run_align_hands
    from v2d.wilor.docker.run_tracks_from_wilor_masks import run_tracks_from_wilor_masks
    from v2d.wilor.docker.run_tracks_interpolate import run_tracks_interpolate
    from v2d.wilor.docker.run_video_to_hands import run_video_to_hands

    S, O = Path(a.stereo_dir), Path(a.out_dir)
    O.mkdir(parents=True, exist_ok=True)
    video = str(S / "cam_a_undist.mp4")
    frames = O / "frames"
    if not step("extract frames", frames.is_dir() and any(frames.iterdir())):
        frames.mkdir(exist_ok=True)
        cap, i = cv2.VideoCapture(video), 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frames / f"{i:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            i += 1
    raw = O / "wilor_raw"
    if not step("WiLoR over all frames", raw.is_dir() and any(raw.iterdir())):
        run_video_to_hands(video_path=video, output_dir=str(raw), weights_dir=WILOR_W)

    hands = [h.split(":") if ":" in h else [h, "auto"] for h in a.hands.split(",")]
    for k, (t, side) in enumerate(hands):
        if side == "auto":  # vote from WiLoR's own handedness on detections overlapping this SAM2 track's mask
            votes = [0, 0]
            for mp in sorted((Path(a.hand_masks) / t).glob("*.png")):
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                rp = raw / f"{mp.stem}.json"
                if m is None or not rp.exists() or not (m > 0).any():
                    continue
                ys, xs = np.nonzero(m > 0)
                mb = (xs.min(), ys.min(), xs.max(), ys.max())
                best, side_best = 0.0, None
                for d in json.load(open(rp)):
                    b = d["bbox"]
                    ix = max(0, min(mb[2], b["x1"]) - max(mb[0], b["x0"]))
                    iy = max(0, min(mb[3], b["y1"]) - max(mb[1], b["y0"]))
                    inter = ix * iy
                    union = (mb[2] - mb[0]) * (mb[3] - mb[1]) + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter
                    if union > 0 and inter / union > best:
                        best, side_best = inter / union, d["is_right"]
                if best > 0.3:
                    votes[int(bool(side_best))] += 1
            side = "right" if votes[1] > votes[0] else "left"
            if min(votes) > 0.6 * max(votes):  # near-tie: an ego left hand sits in the left half of the image
                xs = [np.nonzero(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0)[1].mean()
                      for mp in sorted((Path(a.hand_masks) / t).glob("*.png"))[::5]
                      if (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0).any()]
                if xs:
                    w = cv2.imread(str(next((Path(a.hand_masks) / t).glob("*.png"))), cv2.IMREAD_GRAYSCALE).shape[1]
                    side = "left" if np.mean(xs) < w / 2 else "right"
                    print(f"track {t}: near-tie vote, mask centre x {np.mean(xs):.0f}/{w} -> {side}")
            hands[k] = [t, side]
            print(f"track {t}: WiLoR handedness votes L {votes[0]} / R {votes[1]} -> {hands[k][1]}")
    tracks = O / "hand_tracks.json"
    json.dump({"reference_frame": 0, "sam2_hand_prompt_source": "box",
               "tracks": [{"object_id": int(t), "role": "hand", "is_right": side == "right", "score": 1.0}
                          for t, side in hands]}, open(tracks, "w"), indent=2)

    obj_union = None
    obj_dirs = [Path(d) for d in a.object_masks.split(",") if d]
    if obj_dirs:
        obj_union = O / "object_masks_union"
        if not step("union of object masks", obj_union.is_dir() and any(obj_union.iterdir())):
            obj_union.mkdir(exist_ok=True)
            for f in sorted(frames.glob("*.jpg")):
                m = None
                for d in obj_dirs:
                    x = cv2.imread(str(d / f"{f.stem}.png"), cv2.IMREAD_GRAYSCALE)
                    if x is not None:
                        m = (x > 0) if m is None else (m | (x > 0))
                if m is not None:
                    cv2.imwrite(str(obj_union / f"{f.stem}.png"), m.astype(np.uint8) * 255)

    matched = O / "wilor"
    if not step("match WiLoR detections -> SAM2 hand tracks", matched.is_dir() and any(matched.rglob("*.json"))):
        run_tracks_from_wilor_masks(frames_dir=str(frames), wilor_raw_dir=str(raw), masks_dir=a.hand_masks,
                                    tracks_path=str(tracks), output_dir=str(matched), mano_assets_root=MANO,
                                    min_iou=a.min_iou)
    aligned = O / "wilor_aligned"
    if not step("depth-align hands (stereo depth, real intrinsics)", aligned.is_dir() and any(aligned.rglob("*.json"))):
        run_align_hands(hamer_dir=str(matched), depth_dir=a.depth_dir or str(S / "stereo_depth"),
                        intrinsics_path=str(S / "intrinsics_undist.json"), mano_assets_root=MANO,
                        output_dir=str(aligned), hand_masks_dir=a.hand_masks,
                        object_masks_dir=str(obj_union) if obj_union else None, mask_min_pixels=a.mask_min_pixels)
    # Merge per SIDE before gap filling: run_tracks_interpolate assumes <= 1 track per side and otherwise "tiebreaks"
    # by relabelling same-side tracks as the other hand. Per frame the first track (in prompt order) with a real
    # aligned record wins; masks are the union of the side's tracks.
    n = len(list(frames.glob("*.jpg")))
    side_al, side_mk = O / "side_aligned", O / "side_masks"
    for root in (side_al, side_mk):
        if root.exists():
            shutil.rmtree(root)
    for sid, side in ((0, "left"), (1, "right")):
        ids = [t for t, s in hands if s == side]
        (side_al / str(sid)).mkdir(parents=True)
        (side_mk / str(sid)).mkdir(parents=True)
        n_real = 0
        for f in sorted(frames.glob("*.jpg")):
            pick = next((t for t in ids if (aligned / t / f"{f.stem}.json").exists()), None)
            if pick is not None:
                shutil.copy(aligned / pick / f"{f.stem}.json", side_al / str(sid) / f"{f.stem}.json")
                n_real += 1
            m = None
            for t in ids:
                x = cv2.imread(str(Path(a.hand_masks) / t / f"{f.stem}.png"), cv2.IMREAD_GRAYSCALE)
                if x is not None:
                    m = (x > 0) if m is None else (m | (x > 0))
            if m is not None:
                cv2.imwrite(str(side_mk / str(sid) / f"{f.stem}.png"), m.astype(np.uint8) * 255)
        print(f"{side}: tracks {ids} -> {n_real}/{n} frames with a real aligned record")
    filled = O / "side_filled"
    if not step("interpolate short gaps (one track per side)", filled.is_dir() and any(filled.rglob("*.json"))):
        run_tracks_interpolate(aligned_dir=str(side_al), masks_dir=str(side_mk), output_dir=str(filled),
                               betas="fixed", max_gap_frames=a.max_gap)
    by_side = O / "hands_by_side"
    for sid, side in ((0, "left"), (1, "right")):
        d = by_side / side
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(filled / str(sid), d)
        print(f"{side}: {len(list(d.glob('*.json')))}/{n} frames after filling -> {d}")
    hj = filled / "handedness.json"
    if hj.exists():
        hd = json.load(open(hj))
        for sid, side in ((0, "left"), (1, "right")):
            v = hd.get(str(sid), {})
            if v and bool(v.get("is_right")) != (side == "right"):
                print(f"WARNING {side} track: interpolation vote disagrees ({v.get('votes_right')}R/{v.get('votes_left')}L)")

if __name__ == "__main__":
    main()
