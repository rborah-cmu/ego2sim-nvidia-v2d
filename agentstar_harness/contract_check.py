"""contract_check.py — the Ego2Sim acceptance contract for a shape-only run.

The run's builder agent does not grade itself: this script reads the harness's own artifacts and
decides. It is mounted read-only (harness/), and its spec RUN_DIR/contract_spec.json is pinned
read-only by the sandbox launcher.

    micromamba run -n artscript env PYTHONPATH=harness python -m analysis.contract_check --run-dir RUN_DIR

Checks, on the newest pass (iterations/*/renders/*/report.json):
  silhouette  median gate IoU >= iou_median_min and every frame >= iou_min
  size        SCALE x longest canonical extent of mesh/object.glb within size_tolerance of the stereo target
  watertight  mesh/watertight_report.json exists, is newer than mesh/object.glb, and every part passes
  review      NOTES.md has the line  CONTRACT-REVIEW iteration=<newest> turntable=coherent side_by_side=match
  budget      iteration count <= max_iterations

  surface     median depth error of the newest pass against the capture's metric depth: advisory, a check only
              when the spec sets surface_mae_max_m
  plateau     the last plateau_window (3) full passes did not beat the earlier best by plateau_gain (0.005), after
              at least plateau_min_passes (5); ends the run only when size and watertight pass (plateau_window 0 = off)

Writes RUN_DIR/contract_status.json. Exit 0 = all pass (finalize and stop), 1 = keep working (each failed check
prints what to do), 3 = budget exhausted, 4 = plateau (both: finalize the best accepted state now and report honestly).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def newest_report(run_dir):
    reps = sorted(glob.glob(os.path.join(run_dir, "iterations", "*", "renders", "*", "report.json")))
    return reps[-1] if reps else None


def glb_longest_extent(path):
    import trimesh
    scene = trimesh.load(path, force="scene")
    lo, hi = scene.bounds
    return float(np.max(hi - lo))


def pass_history(run_dir, min_frames=3):
    """[(report path, median IoU, min IoU)] for every full pass, oldest first (sweeps and partial passes are skipped)."""
    out = []
    for rp in sorted(glob.glob(os.path.join(run_dir, "iterations", "*", "renders", "*", "report.json"))):
        v = [f["iou"] for f in json.load(open(rp)).get("frames", []) if f.get("iou") is not None]
        if len(v) >= min_frames:
            out.append((rp, float(np.median(v)), float(min(v))))
    return out


def plateau_reached(history, spec):
    """A sentence when the last `plateau_window` full passes did not beat the best earlier pass by `plateau_gain`
    (after at least `plateau_min_passes` passes), else None. Past runs kept spending their whole budget on a
    silhouette threshold they could not reach; the silhouette score does not predict the mesh score (CD-O)."""
    window = int(spec.get("plateau_window", 3))
    gain = float(spec.get("plateau_gain", 0.005))
    if window <= 0 or len(history) < max(int(spec.get("plateau_min_passes", 5)), window + 1):
        return None
    before = max(h[1] for h in history[:-window])
    recent = max(h[1] for h in history[-window:])
    if recent - before > gain:
        return None
    return (f"the last {window} full passes did not improve the median IoU by more than {gain} "
            f"(best before {before:.3f}, best since {recent:.3f}).")


def surface_line(rep, spec):
    """Depth agreement of the newest pass against the capture's metric depth: (ok, message). Advisory unless the
    spec sets surface_mae_max_m. A large error with a one-sided bias is a pose that is off along the camera ray,
    which hides the shape errors a silhouette cannot see (hollow vs solid, concave faces)."""
    fr = [f for f in rep.get("frames", []) if f.get("depth_mae") is not None]
    if not fr:
        return None
    mae = float(np.median([f["depth_mae"] for f in fr]))
    scale = float(rep.get("scale") or 0)
    bias = float(np.median([f.get("depth_bias_canon") or 0.0 for f in fr])) * scale
    limit = spec.get("surface_mae_max_m")
    ok = limit is None or mae <= float(limit)
    msg = f"median depth error {mae * 100:.1f} cm, median bias {bias * 100:+.1f} cm"
    msg += f" (need <= {float(limit) * 100:.1f} cm)" if limit is not None else " (advisory)"
    if abs(bias) > 0.5 * mae and mae > 0.015:
        msg += (" -> the model sits off along the camera ray in most frames: re-pose with depth, "
                "utils.sweep_batch --frames <all> --depth-weight 0.5 --apply, before judging the shape")
    elif mae > 0.015:
        msg += " -> surfaces disagree with the measured depth: check hollow/open parts and concave faces in depth_residual_<frame>.png"
    return ok, msg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    run = a.run_dir
    spec = json.load(open(os.path.join(run, "contract_spec.json")))
    checks = {}

    iters = sorted(d for d in glob.glob(os.path.join(run, "iterations", "*")) if os.path.isdir(d))
    n_iter = len(iters)
    rep_path = newest_report(run)
    newest_iter = os.path.basename(rep_path.split(os.sep + "renders")[0]) if rep_path else None

    # silhouette
    if rep_path is None:
        checks["silhouette"] = (False, "no pass yet: run harness/utils/shape_pass.sh RUN_DIR --frames <all>")
        rep = {}
    else:
        rep = json.load(open(rep_path))
        ious = {f["frame"]: f.get("iou") for f in rep.get("frames", []) if f.get("iou") is not None}
        if not ious:
            checks["silhouette"] = (False, f"{rep_path} has no per-frame IoU")
        else:
            vals = np.array(list(ious.values()))
            med, worst = float(np.median(vals)), min(ious, key=ious.get)
            ok = bool(med >= spec["iou_median_min"] and vals.min() >= spec["iou_min"])
            msg = f"median {med:.3f} (need >= {spec['iou_median_min']}), worst {worst} {ious[worst]:.3f} (need >= {spec['iou_min']})"
            if not ok:
                msg += (" -> open side_by_side for the worst frames; if a coherent rigid offset, fix that frame's pose "
                        "(one directed sweep); if contours disagree in several frames, fix build()")
            checks["silhouette"] = (ok, msg)

    # size
    glb = os.path.join(run, "mesh", "object.glb")
    if not os.path.isfile(glb) or not rep:
        checks["size"] = (False, "no mesh/object.glb yet (written by shape_pass)")
    else:
        longest = rep.get("scale", 0) * glb_longest_extent(glb)
        tgt, tol = spec["target_longest_m"], spec["size_tolerance"]
        ok = bool(abs(longest / tgt - 1) <= tol)
        msg = f"longest extent {longest*100:.1f} cm vs stereo target {tgt*100:.1f} cm (+/-{tol:.0%})"
        if not ok:
            msg += (f" -> set SCALE = {tgt:.4f} / (canonical longest extent) ; the target is measured from metric stereo "
                    "and is not yours to re-estimate")
        checks["size"] = (ok, msg)

    # watertight
    wr = os.path.join(run, "mesh", "watertight_report.json")
    cmd = ("micromamba run -n artscript env PYTHONPATH=harness python -m analysis.check_watertight "
           "RUN_DIR/mesh/object.glb --out RUN_DIR/mesh/watertight_report.json")
    if not os.path.isfile(wr) or (os.path.isfile(glb) and os.path.getmtime(wr) < os.path.getmtime(glb)):
        checks["watertight"] = (False, f"missing or stale -> run: {cmd}")
    else:
        w = json.load(open(wr))
        bad = [p.get("name", "?") for p in w.get("parts", []) if not p.get("is_watertight", p.get("watertight", False))]
        ok = not bad and bool(w.get("parts"))
        checks["watertight"] = (ok, "all parts watertight" if ok else
                                f"not watertight: {', '.join(bad)} -> fix those parts in build() (separate closed solids)")

    # review (the builder's own visual read, bound to the newest iteration)
    notes = open(os.path.join(run, "NOTES.md")).read() if os.path.isfile(os.path.join(run, "NOTES.md")) else ""
    pat = re.compile(rf"CONTRACT-REVIEW\s+iteration={re.escape(newest_iter or 'none')}\s+turntable=coherent\s+side_by_side=match")
    ok = bool(newest_iter) and bool(pat.search(notes))
    checks["review"] = (ok, "visual review recorded for the newest iteration" if ok else
                        f"after reading the newest turntable sheet and the diagnostic side_by_sides, append to NOTES.md: "
                        f"CONTRACT-REVIEW iteration={newest_iter} turntable=coherent side_by_side=match "
                        "(only if true; otherwise fix the defect)")

    budget_ok = n_iter <= spec["max_iterations"]
    checks["budget"] = (budget_ok, f"{n_iter} iterations used of {spec['max_iterations']}")

    surface = surface_line(rep, spec) if rep else None
    if surface is not None and spec.get("surface_mae_max_m") is not None:
        checks["surface"] = surface

    passed = all(ok for k, (ok, _) in checks.items() if k != "budget")
    history = pass_history(run)
    best = max(history, key=lambda h: h[1]) if history else None
    plateau = None if passed else plateau_reached(history, spec)
    # a plateau ends the run only when everything that is cheap to fix already passes
    plateau_stop = bool(plateau and checks["size"][0] and checks["watertight"][0])
    status = {"newest_iteration": newest_iter, "report": rep_path, "all_pass": passed, "plateau": plateau_stop,
              "best_pass": None if best is None else {"report": best[0], "iou_median": best[1], "iou_min": best[2]},
              "surface": None if surface is None else surface[1],
              "checks": {k: {"passes": ok, "detail": m} for k, (ok, m) in checks.items()}}
    with open(os.path.join(run, "contract_status.json"), "w") as f:
        json.dump(status, f, indent=2)
    for k, (ok, m) in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {k:10s} {m}")
    if surface is not None and "surface" not in checks:
        print(f"[info] {'surface':10s} {surface[1]}")
    if passed:
        print("CONTRACT MET -> finalize: write the final report section in NOTES.md and stop.")
        return 0
    if best:
        print(f"best pass by median IoU: {best[0]} (median {best[1]:.3f}, worst {best[2]:.3f}); its scene is "
              "iterations/<N>/scene.py")
    if not budget_ok:
        print("BUDGET EXHAUSTED -> finalize the best accepted state now and report what still fails.")
        return 3
    if plateau_stop:
        print(f"PLATEAU -> {plateau} Size and watertightness pass, so more passes are unlikely to pay for themselves: "
              "restore the best pass's scene.py if the newest is worse, finalize, report the remaining gap honestly, stop.")
        return 4
    if plateau:
        print(f"note: {plateau} Fix the failing size/watertight check; the run can then end on the best pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
