"""sweep_batch.py — pose sweeps for several frames as ONE bookkept iteration, optionally pasted into scene.py.

    micromamba run -n artscript env PYTHONPATH=harness python -m utils.sweep_batch --run-dir RUN_DIR \
        --frames 000030.jpg,000070.jpg [--ranges "roll:-30,30;yaw:-30,30;pitch:-30,30;dpx:-0.05,0.05;dpy:-0.05,0.05;tz:-0.6,0.6"] \
        [--depth-weight 0] [--quality 0.5] [--jobs 4] [--apply] [--min-gain 0.0]

Each frame runs the normal `sweep` view (harness/views/sweeps/sweep.md) against its object mask and occluder mask,
in parallel, inside a single iteration (`bookkeeping run --kind custom`), so N frames cost one iteration, not N.
Prints one row per frame (start -> best gate IoU, the winning order) and writes ITERATION_DIR/sweep_batch.json.
"start" is the frame's gate IoU in the newest full shape_pass. A DOF that ends within 10% of its range bound is
flagged AT-BOUND: the true correction may lie outside the window, so widen --ranges for that frame and re-run.

--apply (after the iteration has closed) rewrites FRAMES[frame]["pose"] in RUN_DIR/scene.py with each winner whose
gate IoU beats its start by more than --min-gain (non-reference frames get "moved": True; a backup of scene.py is
kept in the iteration dir). Then run shape_pass and LOOK at the side-by-sides before accepting the poses.
"""
import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

DEFAULT_RANGES = "roll:-30,30;yaw:-30,30;pitch:-30,30;dpx:-0.05,0.05;dpy:-0.05,0.05;tz:-0.6,0.6"
HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(HARNESS)


def sweep_one(frame, a, layout, it_dir):
    stem = os.path.splitext(frame)[0]
    cap = layout["capture"]
    cmd = [os.path.join(HARNESS, "render.sh"), os.path.join(a.run_dir, "scene.py"), os.path.join(it_dir, "renders"),
           "--views", "sweep", "--tracking", os.path.join(cap, "tracking"), "--ref-frame", layout["ref_frame"],
           "--frame", frame, "--match-res", os.path.join(layout["frames_dir"], frame),
           "--sweep-mask", os.path.join(layout["masks_dir"], stem + ".png"),
           "--sweep-depth-weight", str(a.depth_weight), "--sweep-quality", str(a.quality), "--sweep-ranges", a.ranges]
    hand = os.path.join(layout.get("hand_masks_dir") or "", stem + ".png")
    if layout.get("hand_masks_dir") and os.path.isfile(hand):
        cmd[cmd.index("--sweep-depth-weight"):cmd.index("--sweep-depth-weight")] = ["--sweep-hand-mask", hand]
    log = os.path.join(it_dir, f"sweep_{stem}.log")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT).returncode
    text = open(log).read()
    m = re.findall(r"PASS \d+ OUTPUT DIR: (\S+)", text)
    sj = os.path.join(m[-1], "sweep.json") if m else None
    if rc or not sj or not os.path.isfile(sj):
        return {"frame": frame, "ok": False, "log": log, "rc": rc}
    d = json.load(open(sj))
    gate = d.get("gate_field", "iou_visible")
    start = (d.get("pose_start") or {}).get(gate) if isinstance(d.get("pose_start"), dict) else None
    if start is None:  # the unchanged pose is the zero order; take its score if the sweep reported it
        for r in d.get("ranked", []):
            if all(abs(v) < 1e-9 for v in (r.get("order") or {}).values()):
                start = r.get(gate)
                break
    b = d["best"]
    return {"frame": frame, "ok": True, "gate": gate, "start": start, "best": b.get(gate), "order": b.get("order"),
            "pose": b["pose"], "sweep_json": sj, "log": log}


def newest_pass_ious(run_dir):
    import glob
    reps = sorted(glob.glob(os.path.join(run_dir, "iterations", "*", "renders", "*", "report.json")))
    if not reps:
        return {}
    return {f["frame"]: f.get("iou") for f in json.load(open(reps[-1])).get("frames", [])}


def at_bound(order, ranges):
    out = []
    for part in ranges.split(";"):
        if ":" not in part:
            continue
        k, b = part.split(":", 1)
        lo, hi = (float(x) for x in b.split(","))
        v = (order or {}).get(k.strip())
        if v is not None and (v >= hi - 0.1 * (hi - lo) / 2 or v <= lo + 0.1 * (hi - lo) / 2):
            out.append(k.strip())
    return out


def fmt_frames(frames):
    lines = ["FRAMES = {"]
    for k, v in frames.items():
        lines.append(f"    {k!r}: {_fmt(v)},")
    lines.append("}")
    return "\n".join(lines)


def _fmt(v):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_fmt(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".") if v != 0 else "0.0"
    return repr(v)


def read_frames(src):
    """FRAMES as a dict, resolving module-level constant names it refers to (e.g. a shared pose tuple)."""
    tree = ast.parse(src)
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                consts[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "FRAMES" for t in n.targets))

    class Inline(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in consts:
                return ast.copy_location(ast.Constant(value=None), n) if consts[n.id] is None else                     ast.copy_location(ast.parse(repr(consts[n.id]), mode="eval").body, n)
            raise ValueError(f"FRAMES refers to {n.id!r}, which is not a module-level literal")

    value = Inline().visit(ast.parse(ast.unparse(node.value), mode="eval").body)
    return node, ast.literal_eval(value)


def apply_poses(scene_path, results, ref, min_gain):
    src = open(scene_path).read()
    node, frames = read_frames(src)
    applied = []
    for r in results:
        if not r["ok"] or r["best"] is None:
            continue
        if r["start"] is not None and r["best"] - r["start"] <= min_gain:
            continue
        entry = dict(frames.get(r["frame"], {}))
        entry["pose"] = {"quaternion": tuple(r["pose"]["quaternion"]), "translation": tuple(r["pose"]["translation"])}
        if r["frame"] != ref:
            entry["moved"] = True
        frames[r["frame"]] = entry
        applied.append(r["frame"])
    lines = src.splitlines()
    out = "\n".join(lines[:node.lineno - 1] + [fmt_frames(frames)] + lines[node.end_lineno:]) + "\n"
    ast.parse(out)  # never leave scene.py unparsable
    open(scene_path, "w").write(out)
    return applied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ranges", default=DEFAULT_RANGES)
    ap.add_argument("--depth-weight", type=float, default=0.0)
    ap.add_argument("--quality", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.0)
    a = ap.parse_args(argv)
    a.run_dir = os.path.abspath(a.run_dir)

    if os.environ.get("HARNESS_ITERATION_DIR"):  # inside the bookkept iteration: sweep only, never touch scene.py
        it_dir = os.environ["HARNESS_ITERATION_DIR"]
        layout = json.load(open(os.path.join(a.run_dir, "layout.json")))
        frames = [f.strip() for f in a.frames.split(",") if f.strip()]
        with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            results = list(ex.map(lambda f: sweep_one(f, a, layout, it_dir), frames))
        json.dump(results, open(os.path.join(it_dir, "sweep_batch.json"), "w"), indent=1)
        return 0 if any(r["ok"] for r in results) else 1

    if a.apply:  # fail BEFORE spending an iteration if FRAMES cannot be rewritten
        try:
            read_frames(open(os.path.join(a.run_dir, "scene.py")).read())
        except (ValueError, SyntaxError, StopIteration) as e:
            print(f"--apply impossible: cannot read scene.py FRAMES ({e}); fix scene.py or drop --apply. No sweep run.")
            return 2
    start = newest_pass_ious(a.run_dir)
    inner = [x for x in (argv if argv is not None else sys.argv[1:]) if x != "--apply"]
    if "--run-dir" in inner:  # the inner run has cwd=harness/, so a relative path would resolve wrongly
        inner[inner.index("--run-dir") + 1] = a.run_dir
    cmd = [sys.executable, "-m", "bookkeeping", "run", "--run-dir", a.run_dir, "--kind", "custom",
           "--label", "batch-sweep", "--", sys.executable, "-m", "utils.sweep_batch"] + inner
    proc = subprocess.run(cmd, cwd=HARNESS, env=dict(os.environ, PYTHONPATH=HARNESS), capture_output=True, text=True)
    try:
        it_dir = json.loads(proc.stdout[proc.stdout.index("{"):])["path"]
    except (ValueError, KeyError):
        print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
        return proc.returncode or 1
    results = json.load(open(os.path.join(it_dir, "sweep_batch.json")))
    for r in results:
        r["start"] = start.get(r["frame"])
    print(f"iteration {os.path.basename(it_dir)} (batch-sweep, {len(results)} frames)")
    print(f"{'frame':12s} {'start':>6s} {'best':>6s}  order")
    for r in results:
        if not r["ok"]:
            print(f"{r['frame']:12s} FAILED rc={r['rc']} (log {r['log']})")
            continue
        s0 = "  -  " if r["start"] is None else f"{r['start']:.3f}"
        order = ", ".join(f"{k} {v:+.3g}" for k, v in (r["order"] or {}).items())
        bound = at_bound(r["order"], a.ranges)
        print(f"{r['frame']:12s} {s0:>6s} {r['best']:.3f}  {order}" + (f"   AT-BOUND: {','.join(bound)} -> widen --ranges" if bound else ""))
    if a.apply:
        import shutil
        shutil.copy2(os.path.join(a.run_dir, "scene.py"), os.path.join(it_dir, "scene_before_apply.py"))
        layout = json.load(open(os.path.join(a.run_dir, "layout.json")))
        applied = apply_poses(os.path.join(a.run_dir, "scene.py"), results, layout["ref_frame"], a.min_gain)
        print(f"applied {len(applied)} pose(s) to scene.py FRAMES: {', '.join(applied) or 'none'}"
              " (only winners that beat their start) -> run shape_pass and look before accepting")
    json.dump(results, open(os.path.join(it_dir, "sweep_batch.json"), "w"), indent=1)
    return 0 if all(r["ok"] for r in results) else 1

if __name__ == "__main__":
    sys.exit(main())
