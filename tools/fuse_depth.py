"""Fill stereo-depth holes with MoGe depth rescaled to the stereo per frame (v2d inverse-depth PNGs in and out).

SGBM stereo (tools/ego_stereo_depth.py) is metric and accurate but covers only ~2/3 of cam_a: the right strip of the head
camera is outside the left mono camera's view. Hands there got no depth, and the hand depth-alignment then kept WiLoR's
unscaled camera translation (ep12 right hand: 54/181 frames at z ~ 3.5 cm). Per frame: k = median(stereo / moge) over
pixels valid in both (dynamic ranges only); output = stereo where valid, else k * moge.

Usage: python fuse_depth.py STEREO_DEPTH_DIR MOGE_DEPTH_DIR OUT_DIR
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def decode(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = 65535.0 / np.maximum(raw, 1.0) - 1.0
    d[raw >= 65535] = 0
    return d


def encode(d):
    return np.clip(65535.0 / (d + 1.0), 0, 65535).astype(np.uint16)


def main():
    S, M, O = (Path(p) for p in sys.argv[1:4])
    O.mkdir(parents=True, exist_ok=True)
    ks, fills = [], []
    for f in sorted(S.glob("*.png")):
        ds = decode(f)
        mp = M / f.name
        if not mp.exists():
            cv2.imwrite(str(O / f.name), encode(ds))
            continue
        dm = decode(mp)
        if dm.shape != ds.shape:
            dm = cv2.resize(dm, (ds.shape[1], ds.shape[0]), interpolation=cv2.INTER_NEAREST)
        both = (ds > 0.1) & (ds < 3.0) & (dm > 0.1)
        k = float(np.median(ds[both] / dm[both])) if both.sum() > 1000 else 1.0
        hole = (ds <= 0) & (dm > 0)
        out = ds.copy()
        out[hole] = k * dm[hole]
        cv2.imwrite(str(O / f.name), encode(out))
        ks.append(k)
        fills.append(hole.mean())
    print(f"{len(ks)} frames fused: per-frame moge scale median {np.median(ks):.3f} (p10 {np.percentile(ks,10):.3f}, "
          f"p90 {np.percentile(ks,90):.3f}); filled {np.mean(fills):.0%} of pixels on average -> {O}")


if __name__ == "__main__":
    main()
