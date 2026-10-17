"""Why does a rollout fail? Per object, from an official evaluator parquet (achieved + reference poses per frame) and the
reference ManoSharpaData parquet (robot wrists):
  - reference motion: when the object starts/stops moving, max lift
  - achieved: max lift, frame where pos error first exceeds 5 cm, whether it ever moved with the reference
  - reference wrist -> object distance while the reference object moves (is a hand actually on it?)
  - verdict: NO_GRASP (achieved never follows), DROP (follows, then loses it), KNOCKED (moves but not lifted),
    or OK.
  diagnose_rollout.py OFFICIAL.parquet [--every 25]
"""
import argparse
import glob

import numpy as np
import pyarrow.parquet as pq


def arr(x):
    return np.asarray(x.to_pylist() if hasattr(x, "to_pylist") else x, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("official")
    ap.add_argument("--every", type=int, default=25)
    a = ap.parse_args()
    t = pq.read_table(a.official)
    meta = {k.decode(): v.decode() for k, v in t.schema.metadata.items()}
    d = t.to_pandas()
    refp = meta.get("flash_chord.reference_parquet", "")
    ref = None
    if refp:
        fs = glob.glob(refp) if refp.endswith(".parquet") else glob.glob(refp + "/*.parquet")
        if fs:
            ref = pq.read_table(fs[0]).to_pylist()[0]
    wr = {}
    if ref is not None:
        for s in ("left", "right"):
            w = np.asarray(ref[f"robot_{s}_wrist_position"], dtype=float)
            wr[s] = w.reshape(-1, 3)
    for name, g in d.groupby("object_body_name", observed=True):
        g = g.sort_values("step")
        A = g[["achieved_position_x", "achieved_position_y", "achieved_position_z"]].to_numpy()
        R = g[["reference_position_x", "reference_position_y", "reference_position_z"]].to_numpy()
        fr = g["reference_frame"].to_numpy()
        err = np.linalg.norm(A - R, axis=1) * 100
        rmove = np.linalg.norm(R - R[0], axis=1) * 100
        amove = np.linalg.norm(A - A[0], axis=1) * 100
        rlift, alift = (R[:, 2] - R[0, 2]) * 100, (A[:, 2] - A[0, 2]) * 100
        moving = np.nonzero(rmove > 2)[0]
        print(f"\n== {name}: {len(g)} frames, ref moves {rmove.max():.1f} cm (lift {rlift.max():.1f}), "
              f"achieved moves {amove.max():.1f} cm (lift {alift.max():.1f}); mean err {err.mean():.1f} cm")
        if len(moving):
            m0, m1 = moving[0], moving[-1]
            print(f"   ref object moving frames {fr[m0]}..{fr[m1]}")
        bad = np.nonzero(err > 5)[0]
        print(f"   err>5cm first at frame {fr[bad[0]] if len(bad) else '-'}; fraction of frames >5cm {np.mean(err > 5):.2f}")
        if wr:
            dist = {s: np.linalg.norm(w[np.clip(fr, 0, len(w) - 1)] - R, axis=1) * 100 for s, w in wr.items()}
            if len(moving):
                for s, dd in dist.items():
                    print(f"   ref {s} wrist->object while moving: median {np.median(dd[moving]):.1f} cm, "
                          f"min {dd[moving].min():.1f}")
        # verdict
        follow = (amove > 0.5 * rmove) & (rmove > 2)
        if not len(moving):
            v = "STATIC_REF"
        elif not follow.any():
            v = "NO_GRASP" if amove.max() < 2 else "KNOCKED"
        else:
            lastf = np.nonzero(follow)[0][-1]
            v = "DROP" if (err[lastf:] > 5).mean() > 0.5 and lastf < moving[-1] - 10 else ("OK" if err.mean() < 3 else "PARTIAL")
            print(f"   follows reference on {follow[moving].mean():.2f} of moving frames, last follow frame {fr[lastf]}")
        print(f"   verdict: {v}")
        print("   frame  refΔ  achΔ  refLift achLift  err" + "".join(f"  {s[0]}wrist" for s in wr))
        for i in range(0, len(g), a.every):
            extra = "".join(f"  {dist[s][i]:6.1f}" for s in wr) if wr else ""
            print(f"   {fr[i]:5d} {rmove[i]:5.1f} {amove[i]:5.1f}  {rlift[i]:6.1f} {alift[i]:6.1f} {err[i]:5.1f}{extra}")


if __name__ == "__main__":
    main()
