# Agent Task — Ego2Sim object mesh (shape-only)
You build a **simulation asset**: a metric, watertight, primitive-built mesh of ONE rigid object seen in
an egocentric video. Downstream, a separate tracker (FoundationPose) recovers the object's motion from
the video using your mesh; **your per-frame poses are discarded**. Poses matter only so the renders line
up well enough to judge the shape. Spend your effort on shape.

You do not grade yourself. `analysis.contract_check` does (see **Done**). Work until it exits 0, or 3.

## Deliverables
- `RUN_DIR/scene.py` — `build()`, `SCALE`, `REFERENCE_FRAME`, `JOINTS = []`, `FRAMES`
  (format: [examples/scene_example.py](examples/scene_example.py), poses: [POSE.md](conventions/POSE.md)).
- `RUN_DIR/mesh/object.glb` + `mesh/pose.json` — written by every `shape_pass`.
- `RUN_DIR/mesh/watertight_report.json`, `RUN_DIR/NOTES.md` (terse), `RUN_DIR/wish.md` (tool gaps; always leave one).

## Inputs (from the launch prompt)
`CAPTURE` (frames, object masks, occluder masks, known cameras, **metric stereo depth**), `FRAMES` (first = reference),
`RUN_DIR`, and **`RUN_DIR/contract_spec.json`** (read-only): the acceptance thresholds and the
**metric size target** measured from stereo depth.
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
5. **Geometry before pose, one kind of change per pass.** A pass either edits geometry or edits poses.
6. **Never step blind:** after every pass open at least one `side_by_side_<frame>.png` before
   reading numbers. IoU is shape-blind; a geometry edit is accepted only on side-by-side + turntable.
7. **Cheap looking:** open one image per question; prefer the diagnostic frames; do not reopen an image
   you already read this pass; never open whole directories of images.

## Loop
1. **Decompose (once).** `micromamba run -n artscript env PYTHONPATH=harness python -m analysis.viz.frame_sheet --run-dir RUN_DIR`;
   read the pages; list in `NOTES.md`: parts, proportions, colours, symmetry, which frames the object
   moved in, and **3 diagnostic frames** (clearest view, a profile, a view showing hidden parts).
2. **First build.** Write `scene.py` with every part you saw and the locked `SCALE`. Seed poses from
   `python -m analysis.measure_depth --run-dir RUN_DIR` (see [measure_depth.md](harness/analysis/measure_depth.md)).
   The capture depth is metric stereo, consistent with the locked `SCALE`: a large depth residual means a pose
   (usually translation along the ray) is wrong, not the scale.
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
   No pose windows, no sub-agents, no temporal report: this task does not need them.
6. **Check:** `micromamba run -n artscript env PYTHONPATH=harness python -m analysis.contract_check --run-dir RUN_DIR`.
   Each FAIL line says what to do. Fix the first failing item and go to 3.
   When only `review`/`watertight` fail, do what they say (the review line is your honest statement
   that the newest turntable is coherent and the side-by-sides match; never write it otherwise).

## Done
- `contract_check` exits **0** → append a short final report to `NOTES.md` (parts list, SCALE,
  final IoU median/worst, anything still imperfect) and **stop immediately**. Do not polish further:
  extra passes cost money and do not change the score.
- exits **3** (budget) → finalize the best accepted state, report honestly what fails, stop.
- Context/token warnings are not a reason to stop early.
