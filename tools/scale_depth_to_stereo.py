"""Scale per-frame monocular (MoGe) depth to metric using stereo depth: s_t = median(stereo / mono) over
pixels where both are valid in [0.2, 3] m; optionally a single global scale (median of s_t).
Writes v2d inverse-depth PNGs to OUT and prints the scale statistics.

Usage: python scale_depth_to_stereo.py MONO_DIR STEREO_DIR OUT_DIR [--global]
"""
import argparse
import json
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
    ap = argparse.ArgumentParser()
    ap.add_argument("mono")
    ap.add_argument("stereo")
    ap.add_argument("out")
    ap.add_argument("--global", dest="glob", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.mono).glob("*.png"))
    scales = []
    for f in files:
        m, s = decode(f), decode(Path(a.stereo) / f.name)
        ok = (m > 0.2) & (m < 3) & (s > 0.2) & (s < 3)
        scales.append(float(np.median(s[ok] / m[ok])) if ok.sum() > 1000 else np.nan)
    sc = np.array(scales)
    g = float(np.nanmedian(sc))
    sc = np.where(np.isfinite(sc), sc, g)
    for f, k in zip(files, sc):
        cv2.imwrite(str(out / f.name), encode(decode(f) * (g if a.glob else k)))
    json.dump({"global_scale": g, "per_frame": sc.tolist()}, open(out / "scales.json", "w"))
    print(f"{len(files)} frames: stereo/mono scale median {g:.3f}  p10 {np.percentile(sc, 10):.3f}  "
          f"p90 {np.percentile(sc, 90):.3f}  ({'global' if a.glob else 'per-frame'} scaling written)")


if __name__ == "__main__":
    main()
