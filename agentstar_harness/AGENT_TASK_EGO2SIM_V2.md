# Agent Task — Ego2Sim object mesh (shape-only, planner + builder)
You build a **simulation asset**: a metric, watertight, primitive-built mesh of ONE rigid object seen in
an egocentric video. Downstream, a separate tracker (FoundationPose) recovers the object's motion from
the video using your mesh; **your per-frame poses are discarded**. Poses matter only so the renders line
up well enough to judge the shape.

The mesh is scored on **surface shape** (Chamfer distance to a scan, after scale is normalised away). In past
runs the silhouette IoU did not predict that score: objects with the lowest IoU scored best, and a run that
only modelled a hollow part as hollow halved its error at the same IoU. So a matching outline is necessary,
not the goal: get the real surfaces right (thin walls, openings, concave faces), and stop when told to.

You do not grade yourself. `analysis.contract_check` does (see **Done**).

## Two roles
- **You are the builder.** You run the loop below: edit `scene.py`, run passes, look, fix, check.
- **`shape-planner`** is a sub-agent on a stronger model. It decides *what to build and why*; you carry it out.
  Call it with the Agent tool (`subagent_type: "shape-planner"`), giving it `RUN_DIR` and the files to read.
  It writes `RUN_DIR/PLAN.md`. Its reasoning does not carry over to you, so `PLAN.md` is the hand-off:
  follow it, and do not re-derive what it already settled.
  1. **Once, after Decompose:** pass the frame-sheet pages and your parts list. It returns the part recipe,
     numeric proportions, which parts are hollow or open, the symmetry, and the diagnostic frames.
  2. **Again only when stuck** (at most 2 more calls), with `NOTES.md` and the 2 side-by-sides that show the problem:
     - `contract_check` prints `PLATEAU` or `note: the last ... passes did not improve`;
     - an edit you believe is right lowered the median IoU by more than 0.03 and a re-pose did not recover it;
     - two views disagree about a part and you cannot tell which is right.
  Everything else is yours. Do not ask the planner to run passes, sweeps or checks.

## Deliverables
- `RUN_DIR/scene.py` — `build()`, `SCALE`, `REFERENCE_FRAME`, `JOINTS = []`, `FRAMES`
  (format: [examples/scene_example.py](examples/scene_example.py), poses: [POSE.md](conventions/POSE.md)).
- `RUN_DIR/mesh/object.glb` + `mesh/pose.json` — written by every `shape_pass`.
- `RUN_DIR/mesh/watertight_report.json`, `RUN_DIR/NOTES.md` (terse), `RUN_DIR/wish.md` (tool gaps; always leave one).
- `RUN_DIR/mesh/symmetry.json` — which rigid motions leave the object looking the same, in canonical axes:
  `{"revolution_axis": "z" | null, "flips_180": ["x", "y", "z"], "note": "..."}` (empty list if none).
  The tracker and the trajectory cleaner use it to tell a real rotation from a look-alike pose.

## Inputs (from the launch prompt)
`CAPTURE` (frames, object masks, occluder masks, known cameras, **metric stereo depth**), `FRAMES` (first = reference),
`RUN_DIR`, and **`RUN_DIR/contract_spec.json`** (read-only): the acceptance thresholds and the
**metric size target** measured from stereo depth.
**Depth.** The capture's depth comes from the camera's own sensor when it has one (a stereo pair or LiDAR), and
`CAPTURE/capture.json` → `source.depth_values` says which. Sensor depth is metric: use it for sizes, heights and for
placing the model along the camera ray. If it says the depth is estimated from a single image, treat it as a hint for
shape only: its scale and its flat or tilted surfaces can be wrong, so lean on the size target in the contract.
`HAND_MASKS_DIR` holds **occluders**: hands, plus other objects only where they are in FRONT of this
object (depth-ordered). Pass it as `--hand-mask`; it is don't-care. Everything else is fair game,
so a render spilling onto a tray or rack behind the object is penalised, as it should be.

## Rules
1. **You own `scene.py` only.** Never create cameras/lights or touch render settings.
   Reuse harness tools; do not read other `runs/`.
2. **Rigid object.** `JOINTS = []`. No articulation, no mechanism work.
3. **Canonical build:** natural orientation, `+Z` up, centred at the origin, longest dimension = 1.0.
   Primitives + booleans + bevels via the shared helpers ([harness/shapes/README.md](harness/shapes/README.md)).
   Separate closed solids per part; colour every part.
4. **Scale is locked:** `SCALE = contract_spec.target_longest_m` (with longest canonical dimension 1.0).
   Do not tune it to raise IoU. If size fails the contract, fix the canonical proportions or SCALE
   arithmetic, not the target.
5. **Real surfaces.** Where the object is a thin shell (bowls, cups, trays, pans, channels), model a shell of the
   real wall thickness, open where the object is open. A solid fill where the object is hollow costs score even
   when the IoU is good. Build a hollow body as outer solid minus inner solid from the same profile, so it stays watertight.
6. **Keep the canonical centre fixed.** The build is re-centred on its bounding box, so a part that grows the
   bounding box shifts the whole mesh under the fixed poses and every frame's IoU drops. After any edit that
   changes the extents, re-pose all frames (one `sweep_batch`, below) before judging the edit, and do not
   revert a correct shape edit because the IoU fell on the first pass.
7. **Geometry before pose, one kind of change per pass.** A pass either edits geometry or edits poses.
8. **Never step blind:** after every pass open at least one `side_by_side_<frame>.png` before
   reading numbers. A geometry edit is accepted only on side-by-side + turntable.
9. **Cheap looking:** open one image per question; prefer the diagnostic frames; do not reopen an image
   you already read this pass; never open whole directories of images.
10. **Model the object, not the capture rig.** Leave out anything attached to the object for recording: motion-capture
    marker balls and their stems, tape, stickers, labels, cables. They are not part of the object, the reference
    scan does not have them, and each one modelled costs score. Note in `NOTES.md` what you left out.
11. **Measure heights, do not estimate them.** A head camera looks down, so heights and relief are what it sees worst
    and what past meshes got most wrong (a pan built half its real height, a handle arch a third too low). Take every
    out-of-plane dimension from the depth, not from the outline: total height, wall height, handle and rim heights,
    how far a foot or bulge sticks out underneath.
    - Start from the fused oriented-box extents that `analysis.measure_depth` reports (W:H:D).
    - For a part, read the depth across it in a frame that sees it from the side, or compare the depth of its top
      with the surface it stands on.
    - **The underside counts.** While the object rests on the table its underside is hidden; the frames where it is
      lifted or tilted are the only evidence. Make one of the diagnostic frames such a frame, and model what it
      shows (feet, bulges, recesses, a raised base) rather than a flat bottom.
    - If a height cannot be measured in any frame, say so in `NOTES.md` and use the planner's estimate.
12. **Undo what does not help.** Keep an edit only if the side-by-sides and the depth agreement are at least as good
    as before it. Otherwise restore the previous `scene.py` and try something else: do not stack edits on a worse state.

## Loop
1. **Decompose (once).** `micromamba run -n artscript env PYTHONPATH=harness python -m analysis.viz.frame_sheet --run-dir RUN_DIR`;
   read the pages; list in `NOTES.md`: parts, colours, which frames the object moved in. Then call
   `shape-planner` (role 1) and read `PLAN.md`.
2. **First build.** Write `scene.py` from `PLAN.md` with the locked `SCALE`. Seed poses from
   `python -m analysis.measure_depth --run-dir RUN_DIR` (see [measure_depth.md](harness/analysis/measure_depth.md)).
3. **Pass:** `harness/utils/shape_pass.sh RUN_DIR --frames <all>` ([shape_pass.md](harness/utils/shape_pass.md)).
4. **Look:** side-by-sides of the 3 diagnostic frames + the 2 worst-IoU frames, then the turntable sheet.
   In `NOTES.md`, one line per frame: shape 0–1, pose 0–1, the defect.
5. **Route the biggest visible defect:**
   - *shape* (contours/proportions/missing part disagree in 2+ views) → edit `build()`.
   - *pose* (whole object offset/rotated, shape reads right) → ONE batch sweep over just those frames, which
     costs one iteration and pastes the winners into `FRAMES`:
     `micromamba run -n artscript env PYTHONPATH=harness python -m utils.sweep_batch --run-dir RUN_DIR --frames a.jpg,b.jpg --apply`
     (`--help` for ranges/depth weight; search semantics in [sweep.md](harness/views/sweeps/sweep.md)).
     Never hand-roll per-frame sweep loops.
   - *depth* (`contract_check` reports the model off along the camera ray) → the same sweep over all frames with
     `--depth-weight 0.5`. The capture depth is metric stereo, consistent with the locked `SCALE`, so this puts
     the model at the measured distance; only then does `depth_residual_<frame>.png` show shape errors.
   No pose windows and no other sub-agents: this task does not need them.
6. **Check:** `micromamba run -n artscript env PYTHONPATH=harness python -m analysis.contract_check --run-dir RUN_DIR`.
   Each FAIL line says what to do; the `surface` line reports depth agreement. Fix the first failing item and go to 3.
   When only `review`/`watertight` fail, do what they say (the review line is your honest statement
   that the newest turntable is coherent and the side-by-sides match; never write it otherwise).

## Done
- `contract_check` exits **0** → append a short final report to `NOTES.md` (parts list, SCALE,
  final IoU median/worst, depth error, anything still imperfect), write `mesh/symmetry.json`, and **stop immediately**.
- exits **3** (budget) or **4** (plateau) → restore the best pass's `scene.py` if the newest is worse, run one
  final pass, write `mesh/symmetry.json`, report honestly what still fails, stop. Do not polish further:
  extra passes cost money and do not change the score.
- Context/token warnings are not a reason to stop early.
