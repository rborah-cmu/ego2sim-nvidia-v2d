# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "matplotlib", "opencv-python-headless"]
# ///
"""Explore the V2D Track 3 public set.

Usage (run from anywhere; DATA defaults to ../v2d_data/track_3/public next to this script's dir):
  uv run tools/explore_track3.py summary                 # per-episode table -> stdout + out/summary.csv
  uv run tools/explore_track3.py plot [EP ...]           # object xyz / visibility plots -> out/traj_epXXXXXX.png
  uv run tools/explore_track3.py frame EP FRAME          # 3 ego frames + object poses -> out/frame_epXXXXXX_fYYYY.png
  uv run tools/explore_track3.py --data /path/to/track_3/public summary
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "v2d_data" / "track_3" / "public"
OUT = ROOT / "out"
CAMS = ["ego_cam_a", "ego_cam_b", "ego_cam_c"]


def load_meta(data: Path):
    info = json.loads((data / "meta" / "info.json").read_text())
    tasks = {t["task_index"]: t["task"] for t in info["tasks"]}
    return info, tasks


def episode_path(data: Path, ep: int) -> Path:
    return data / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"


def video_path(data: Path, ep: int, cam: str) -> Path:
    return data / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"


def load_episode(data: Path, ep: int):
    """Return (df, objects) where objects[name] = {'pose': (T,7) with NaN when invisible, 'visible': (T,)}."""
    table = pq.read_table(episode_path(data, ep))
    df = table.to_pandas()
    names = [o["name"] for o in df["observation.objects"].iloc[0]]
    objects = {}
    for i, name in enumerate(names):
        vis = np.array([row[i]["visible"] for row in df["observation.objects"]], dtype=bool)
        pose = np.stack([np.asarray(row[i]["pose"], dtype=np.float32) for row in df["observation.objects"]])
        pose[~vis] = np.nan  # zeros when invisible are not a valid pose
        objects[name] = {"pose": pose, "visible": vis}
    return df, objects


def cmd_summary(data: Path, _args):
    info, tasks = load_meta(data)
    rows = []
    for ep in info["episode_indices"]:
        df, objs = load_episode(data, ep)
        dur = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
        for name, o in objs.items():
            p = o["pose"][o["visible"]]
            disp = np.linalg.norm(p[-1, :3] - p[0, :3]) if len(p) else np.nan
            path = np.nansum(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1)) if len(p) > 1 else 0.0
            rows.append(dict(
                episode=ep, task=tasks[int(df["task_index"].iloc[0])], frames=len(df), dur_s=round(dur, 2),
                object=name, visible_pct=round(100 * o["visible"].mean(), 1),
                z_min=round(float(np.nanmin(p[:, 2])), 3) if len(p) else np.nan,
                z_max=round(float(np.nanmax(p[:, 2])), 3) if len(p) else np.nan,
                net_disp_m=round(float(disp), 3), path_len_m=round(float(path), 3),
            ))
    s = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(s.to_string(index=False))


def cmd_plot(data: Path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info, tasks = load_meta(data)
    eps = args.episodes or info["episode_indices"]
    OUT.mkdir(exist_ok=True)
    for ep in eps:
        df, objs = load_episode(data, ep)
        t = df["timestamp"].to_numpy()
        fig, axes = plt.subplots(len(objs), 2, figsize=(12, 3 * len(objs)), squeeze=False)
        for r, (name, o) in enumerate(objs.items()):
            ax = axes[r, 0]
            for k, lab in enumerate("xyz"):
                ax.plot(t, o["pose"][:, k], label=lab)
            ax.set_title(f"{name} position (m)")
            ax.legend(loc="upper right")
            axes[r, 0].fill_between(t, *ax.get_ylim(), where=~o["visible"], color="red", alpha=0.15, step="mid")
            ax2 = axes[r, 1]
            p = o["pose"]
            ax2.plot(p[:, 0], p[:, 1])
            ax2.set_aspect("equal", adjustable="datalim")
            ax2.set_title(f"{name} top-down x/y (m)")
        fig.suptitle(f"episode {ep}: {tasks[int(df['task_index'].iloc[0])]} (red = not visible)")
        fig.tight_layout()
        out = OUT / f"traj_ep{ep:06d}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)
        print(out)


def cmd_frame(data: Path, args):
    import cv2

    _, tasks = load_meta(data)
    df, objs = load_episode(data, args.episode)
    imgs = []
    for cam in CAMS:
        cap = cv2.VideoCapture(str(video_path(data, args.episode, cam)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read frame {args.frame} of {cam} ({n} frames)")
        if args.rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        h = 480
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        cv2.putText(img, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        imgs.append(img)
    panel = np.hstack(imgs)
    lines = [f"ep {args.episode} frame {args.frame}/{len(df)} t={df['timestamp'].iloc[args.frame]:.2f}s "
             f"task={tasks[int(df['task_index'].iloc[args.frame])]}"]
    for name, o in objs.items():
        p = o["pose"][args.frame]
        lines.append(f"{name}: " + ("not visible" if not o["visible"][args.frame]
                                    else "xyz=" + np.array2string(p[:3], precision=3) +
                                    " q(wxyz)=" + np.array2string(p[3:], precision=3)))
    bar = np.zeros((30 * len(lines) + 10, panel.shape[1], 3), np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(bar, ln, (10, 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    OUT.mkdir(exist_ok=True)
    out = OUT / f"frame_ep{args.episode:06d}_f{args.frame:04d}.png"
    cv2.imwrite(str(out), np.vstack([panel, bar]))
    print("\n".join(lines))
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    p = sub.add_parser("plot")
    p.add_argument("episodes", nargs="*", type=int)
    f = sub.add_parser("frame")
    f.add_argument("episode", type=int)
    f.add_argument("frame", type=int)
    f.add_argument("--rotate", action="store_true", help="rotate 180 deg (calib says ego rotation_deg=180, but the mp4s are already upright)")
    args = ap.parse_args()
    {"summary": cmd_summary, "plot": cmd_plot, "frame": cmd_frame}[args.cmd](args.data, args)


if __name__ == "__main__":
    main()
