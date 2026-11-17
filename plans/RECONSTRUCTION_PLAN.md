# Ego2Sim reconstruction: implementation plan

For a coding agent picking this up cold. Read all of it before changing anything. Section 9 is the order of work.

## 1. Goal

Ego2Sim turns a head-mounted video of a person manipulating objects into a simulation scene and a robot policy that
reproduces the object motion. This plan covers the reconstruction half: object mesh, object 6-DoF trajectory, camera
path and hand motion, from video.

**The goal is the most accurate reconstruction we can get from video alone.**

Two rules define "from video alone" and they are not negotiable:

1. **The final pipeline takes video and nothing else.** No motion capture, no scanned meshes, no hand-measured object
   sizes at inference time. A separate video of the same object (a "scan clip", section 5.2) is allowed: it is video.
2. **Motion capture is for measuring and for training, never an input to reconstruction.** We have an OptiTrack rig.
   Its trajectories may be used to (a) score every stage, (b) train or tune models and thresholds that then run on
   video only, and (c) stand in for the object trajectory when training policies, but only until the video
   trajectories pass the gate in section 8. After that, policy training uses reconstructed trajectories only.

## 2. Where things stand

Numbers are from the NVIDIA Video-to-Data Challenge (Track 3) and from our first own recordings. Treat them as the
baseline to beat; re-measure before relying on any of them.

| Stage | Current method | Measured | Main weakness |
|---|---|---|---|
| Depth | FoundationStereo on the challenge's stereo head rig; MoGe on our single phone camera | MoGe is 3-6% short against mocap distance (3 clips) | our own recordings have one camera and uncalibrated intrinsics |
| Object mesh | Claude builds it in Blender, render-and-compare (the "Ego2Sim agent"); depth fusion for one static object | Chamfer distance to scan (CD-O) 0.19-0.87 cm per object, 0.29 on the hidden set | the loop optimises silhouette overlap, which does not predict CD-O (section 2.1) |
| Object tracking | FoundationPose, 16 hypotheses, on metric depth; frame 0 set from the still start | against mocap, 9 public episodes: AUC 0.833, mean error 2.85 cm with scanned meshes; **0.691 with our own meshes** | our meshes; look-alike poses of symmetric objects; frame-0 error rotates a whole scene in the scorer |
| Camera path | stereo visual odometry (challenge); **derived from mocap** on our recordings | box overlap 0.79-0.82 median on our clips | on our recordings it is not video-only yet |
| Hands | WiLoR + SAM 2 masks + depth alignment + gap filling; teammates' multi-view tracker on the challenge rig | 48 mm 3D error with the multi-view tracker; on our phone clips the hand is detected in only 40-60% of frames | detection gaps, depth of the hand, no ground truth on our rig |
| Reference cleaning | de-spin, flip unwrap, settle, push apart overlapping hulls | hidden-set AUC 0.35 -> 0.65 together with better policies | rules were tuned by hand on 20 episodes |

### 2.1 Findings that should shape the work

- **Silhouette overlap is the wrong target for meshes.** Across 10 agent runs, final overlap and CD-O are unrelated
  (rank correlation +0.07). The object with the lowest overlap (0.85) had the best CD-O. Modelling a pitcher as a
  hollow shell took CD-O from 0.87 to 0.64 at the same overlap. Only 1 of 11 runs met the 0.93 overlap gate; the rest
  spent their whole budget on it. Sample is small: confirm on the benchmark (section 4).
- **The agent's poses drift along the camera ray.** Final passes show 2-8 cm median depth error with a one-sided bias,
  because pose sweeps ran with depth weight 0. Until that is fixed the depth residual cannot show shape errors.
- **Our own meshes cost more accuracy than anything else in tracking** (0.833 -> 0.691 AUC).
- **Symmetric and near-symmetric objects flip.** A plain cuboid has 4 equivalent orientations; a spoon read
  upside-down cost a whole episode on the leaderboard. Resolving flips needs both a symmetry declaration per mesh
  and temporal reasoning.
- **A previous video replay tool cut clips at 60%** (frame count vs control rate). Fixed in `scripts/sim_video.sh`.
  Check lengths whenever a tool converts between video frames and simulator steps.

## 3. Code, data and machines

- Repo: `the-simsons/ego2sim-nvidia-v2d`. **Branch from `main`** (it has teammates' hand tracking and policy code).
  There are uncommitted changes in one checkout (tiered agent harness, plateau stop, cost report); ask before
  assuming they are merged.
- `tools/` (reconstruction tools), `scripts/` (drivers), `agentstar_harness/` (agent brief, contract, planner agent),
  `policy_s0/` (policy training). `scripts/ego2sim_public_episode.sh` is the reference driver for one episode.
- Compute: `riddle` (2x RTX 4090, shared lab machine). Read `CLAUDE.md` and `NOTES.md` first: they are not in the
  repo; they live at `~/v2d/` on riddle, mirrored at `D:\v2d\` on the laptop. `NOTES.md` is the running log of what
  was installed, measured and tried, with exact commands. Hard rules:
  never run `sudo`; use a GPU only when it is completely free; never reboot; keep everything under `~/v2d` or
  `/usr1/e2s/rborah_v2d`; before any run over about 30 minutes, state the command, duration and GPU and get a yes.
- Our recordings: `/usr1/e2s/rborah_v2d/aims/` holds 3 clips (`vid/*.MOV`, phone, about 30 fps, one camera) with
  OptiTrack exports (`traj/*.csv`, 240 Hz, Z up, metres, one rigid body `BlueBox`), plus working scripts that are
  **not in the repo yet** (`sync_clips.py`, `fit_camera.py`, `cam_from_fp.py`, `refine_camera.py`, `mocap_table.py`,
  `aims_*.sh`, `check_reference.py`, `replay_measure.py`, `make_aims_package.py`). They hard-code paths and one
  object. Task P0.1 brings them in.
- Challenge data with mocap ground truth: 20 public episodes under `~/v2d/v2d_data/track_3/public`, scans included.
  Scoring tools: `tools/ref_parquet_as_rollout.py`, `tools/score_public_episode.py`, `tools/cdo_board.py`.

## 4. First build the measuring stick (phase P0)

Nothing else in this plan can be judged without it. Deliver a benchmark that scores any reconstruction against
mocap, per stage, with one command.

**P0.1 Bring the mocap tools into the repo** as a package `mocap/` (or `e2s/mocap/`): CSV loader (Motive export:
units, up axis, quaternion order, frame rate from the header), time sync, rigid-body-to-mesh transform, table
height from rest poses. Remove hard-coded paths and the single-object assumption. Keep the scripts' behaviour:
their outputs on the 3 existing clips are the regression test.

**P0.2 Clip manifest.** One JSON per clip: video path(s), intrinsics, mocap file, rigid bodies with their mesh and
transform, sync offset, which hand, split (`train` / `val` / `test`). Splits are by **object and by recording
session**, never by frame, so a model tuned on mocap is always tested on objects and sessions it has not seen.

**P0.3 Metrics**, all computed against mocap after a single rigid alignment of the reconstructed world to the mocap
world, estimated on the first still stretch only (so drift later in the clip is not hidden):

| Stage | Metric |
|---|---|
| Object trajectory | position error (cm) and rotation error (deg): median, 95th percentile, max; AUC and mean per-point error with the challenge code; count and duration of look-alike flips (rotation error within 10 deg of a symmetry of the mesh); frame-0 error |
| Object trajectory, symmetric objects | the same, with rotation error taken modulo the declared symmetry, reported next to the plain numbers |
| Mesh | CD-O against the scan or CAD with the challenge code; longest-extent error (%); watertight yes/no |
| Camera path | needs markers on the camera (P1.1): translation and rotation error, drift per metre |
| Hands | needs a marker cluster on the wrist (P1.1): wrist position and orientation error; fraction of frames with a real detection; hand-to-object surface distance in frames where mocap shows the object moving |
| Timing | sync offset found from video alone vs the offset from a sync event (P1.2) |
| End to end | the reference replayed in simulation with the assist off, and later the trained policy, scored against mocap |

**P0.4 Report.** `bench/run.py --split val --pipeline <config>` writes a table per clip and a summary, and appends
one row to `bench/history.csv` (git commit, config, every summary number). Every later task is accepted or rejected
on this table. Add a `--stage` option so a single stage can be re-scored with the others taken from cache.

**P0.5 The mocap firewall.** Make the rule in section 1 hard to break by accident:
- Reconstruction code lives in a package that must not import the mocap package. Add a test that fails if it does.
- Every output (bundle manifest, reference parquet metadata, package `quality.csv`) carries
  `trajectory_source: video | mocap` and `mocap_used_for: []`. The policy trainer reads it (section 8).
- The benchmark runs reconstruction in a process that cannot see the mocap files (separate directory, passed only to
  the scoring step).

Acceptance for P0: the 3 existing clips score through `bench/run.py`; the existing mocap-assisted outputs reproduce
within 1 mm; a **video-only baseline** exists for those clips (camera from video, object from FoundationPose, table
from depth) with its numbers in `bench/history.csv`. That baseline is expected to be poor. It is the number to beat.

## 5. Accuracy work, in order of expected gain

Each item states what to build, how to judge it, and what would make us drop it. Do them in the order of section 9,
one at a time, each as its own change scored on the benchmark.

### 5.1 Capture (P1): fix what the pipeline cannot recover later

The first recordings used a phone's ultra-wide camera with a guessed focal length, no distortion model, no depth
sensor and no sync event. Fixing capture is cheaper than compensating in software.

- **P1.1 Ground truth for more stages.** Rigid marker clusters on the camera and on the back of each hand or wrist.
  One-time calibration from the camera's marker cluster to its optical centre (hand-eye calibration with a
  checkerboard seen by the camera while mocap tracks the cluster). These are for scoring only (firewall applies).
- **P1.2 Sync.** A sharp tap of the object on the table at the start and end of every take, and a bright LED visible
  to the camera and timestamped by the mocap system if the lab has a sync box. Video-only sync must then be within
  one video frame of it.
- **P1.3 Camera.** Calibrate intrinsics and distortion once per camera and lens (checkerboard, fisheye model if the
  ultra-wide is kept) and undistort before anything else. Prefer a rig with real depth: a calibrated stereo pair
  like the challenge's, or the phone's LiDAR depth recorded with the video. Record at 60 fps with a short exposure:
  motion blur costs tracking during fast moves.
- **P1.4 A recording protocol.** Per object: one scan clip (5.2); per take: 2 s still at the start and end, taps,
  the object set down in a different resting pose at least once.
- **P1.5 More objects.** At least 8 objects for a meaningful benchmark: boxes, a cylinder, a cup, a pan, a tool with
  a handle, something thin, something textureless, something symmetric. Use objects with scans or CAD where possible
  so CD-O can be scored.

Judge by: the same clip scored with and without each capture change (depth sensor vs monocular depth; calibrated
vs guessed intrinsics). Keep monocular-only as a tracked configuration, because it is the hardest and most general.

### 5.2 Object mesh (P2)

Largest single loss today. Three sources, then a selector.

- **P2.1 Scan clip reconstruction (new, expected to be the best source).** A 20-30 s video orbiting the object on
  the table, then flipped and orbited again. Segment with SAM 2, estimate cameras from the background, reconstruct
  (a surface-oriented method: 2D Gaussian splatting or neural SDF, or classical multi-view stereo), merge the two
  halves by registering the object, scale from metric depth, make watertight, simplify. Output in a fixed canonical
  frame with a symmetry file (P2.4). This uses video only, so it is allowed in the final pipeline.
- **P2.2 The agent loop, corrected.** Already drafted in `agentstar_harness/AGENT_TASK_EGO2SIM_V2.md` and
  `contract_check.py`. Finish it:
  - pose sweeps with depth weight on, so the model sits at the measured distance;
  - gate on **surface agreement**, not overlap: per-frame depth residual inside the mask after the pose is fixed,
    reported in millimetres; keep overlap as a sanity floor (for example 0.85), not a target of 0.93;
  - a fixed canonical origin, so a proportion edit does not invalidate every pose;
  - a fitting tool that tunes the numeric parameters of `build()` (proportions, radii, wall thickness) against the
    renders with a derivative-free optimiser, so the model spends turns on structure, not on nudging numbers.
  Run the planner/builder split (`E2S_TIERED=1`) on at least 6 objects and compare cost and CD-O with the
  single-model runs before making it the default.
- **P2.3 Image-to-3D model as a candidate.** One or two current open models (single image or few views to mesh),
  scaled with metric depth. Cheap to add, sometimes good on common objects.
- **P2.4 Symmetry declaration** for every mesh: `{"revolution_axis": ..., "flips_180": [...]}` in the canonical
  frame, written by the source that made the mesh and checked by rendering (a declared symmetry must reproduce the
  silhouettes and depth within noise). Consumed by tracking (5.3) and cleaning.
- **P2.5 A selector that needs no ground truth.** Score every candidate mesh by render-and-compare on the task
  clip (depth residual, silhouette, and photometric error if textured) and pick the best. Then check on the
  benchmark that the selector's choice has the best CD-O; if it does not, fix the score, not the choice.

Judge by: CD-O and longest-extent error per object on `val`, and the object-trajectory metrics when tracking uses
that mesh (a mesh is only as good as the tracking it supports). Target for P2: tracking with our meshes within
0.03 AUC of tracking with scans, on `val`.

### 5.3 Object trajectory (P3)

- **P3.1 Symmetry-aware tracking.** Pass the declared symmetries to FoundationPose, and replace per-frame winner
  selection with a global choice over time: keep the top hypotheses per frame (including symmetric twins) and solve
  for the smoothest consistent path (Viterbi), with terms for the table plane and for hand contact. A prototype
  exists for a cuboid in `cam_from_fp.py` (`pick_flips`).
- **P3.2 Render-and-compare refinement** after tracking: per frame, optimise the pose against depth, silhouette and
  edges (and texture when the mesh has it), with a temporal smoothness term, over a sliding window. Initialise from
  P3.1. This is where sub-centimetre accuracy has to come from.
- **P3.3 Physical constraints.** A resting object lies on the table plane in a stable pose; an object does not pass
  through the table or another object; an object moves only while a hand is near it. Apply as soft terms in P3.2
  first, and keep the existing cleaning stages as a final pass.
- **P3.4 Frame 0.** The first pose anchors everything downstream. Estimate it from the whole still stretch (many
  frames, multi-hypothesis, symmetry resolved using later motion), not from one frame.
- **P3.5 A learned corrector, trained on mocap (allowed use b).** Only after P3.1-P3.4 plateau. A small model that
  takes the tracker's output and image/depth crops and predicts a pose correction and an uncertainty, trained on
  `train` clips with mocap as the label. It runs on video only. Accept it only if it improves `test` objects it has
  never seen; otherwise it has memorised objects and must be dropped.
- **P3.6 Confidence per frame and per clip**, calibrated on mocap: a number that predicts the error without mocap
  (from residuals, hypothesis agreement, occlusion). Needed for section 8: once mocap is gone, this is the only way
  to reject a bad reconstruction.

Judge by: the object-trajectory table on `val`, then `test`. Report flips separately: one flip is worse for policy
training than a centimetre of noise.

### 5.4 Camera path (P4)

Needed to put hands and objects in one world, and it must come from video.

- Visual odometry or SLAM on the static background with hands and objects masked out, metric scale from depth.
  Compare stereo VO (existing `tools/ego_vo.py`), a learned SLAM system, and depth-sensor odometry where available.
- Use the table plane and gravity as constraints: the table is flat and still.
- Jointly refine camera and static objects over the clip (an object that mocap would show as still should be still).
- Score against the camera marker cluster (P1.1). Until P1.1 exists, score indirectly: reconstructed object path in
  the world against mocap.

### 5.5 Hands (P5)

- Start from the teammates' multi-view tracker on a rig with more than one camera; on a single camera, WiLoR or
  HaMeR with metric depth alignment.
- Close detection gaps: tracking across frames, re-detection with a crop guided by the previous pose, and a second
  detector as fallback. Target: a real detection in over 90% of frames where the hand is in view.
- **Contact refinement.** When the object moves, a hand is holding it. Optimise the hand pose so fingertips lie on
  the object surface during those frames and do not penetrate it, keeping the image evidence. The object-carried
  gap fill in `tools/fill_hand_gaps_object.py` is a crude version of this.
- Score against the wrist marker cluster (P1.1) and by hand-to-surface distance during mocap-confirmed motion.
- The object-centric policy tolerates hand error better than hand imitation does, so rank this below 5.2 and 5.3
  unless the benchmark shows hands limiting the policy.

### 5.6 Cost and throughput (P6)

- Record tokens and dollars per agent run (`tools/token_usage.py` prints dollars; the log is root-owned on riddle,
  so the user runs it).
- Keep the cheaper-builder split only if CD-O per dollar improves on the benchmark.
- The plateau stop saved 9% of iterations on past runs: keep it, do not expect more from it.
- Cache every stage by content hash so re-scoring one stage does not re-run the others.

## 6. What mocap may train

Allowed, with the test in brackets:
- thresholds and weights of any stage (re-scored on `test`);
- the confidence model of P3.6 (calibration on `test`);
- the learned pose corrector of P3.5 (must improve unseen objects);
- depth scale and bias corrections for a camera (must hold on a different recording session);
- the mesh selector's score (its choice must match the CD-O ranking on `val`).

Not allowed: any per-clip quantity taken from mocap at inference (object pose, camera pose, table height, sync
offset, object size), and tuning on `test`.

## 7. Risks and how to notice them

- **Overfitting to a few objects.** Split by object and session; report `train`, `val`, `test` side by side.
- **Mocap is not perfect.** Marker-to-mesh transforms are estimated (the box's was fitted from rest poses to
  0.14 mm). Record the residual of every such fit and do not chase errors below it.
- **Alignment hiding errors.** Align worlds on the first still stretch only; report drift.
- **A good benchmark number with a bad policy.** Keep the end-to-end row in P0.3 and run it at every phase boundary.
- **Shared machine.** Long jobs wait for a free GPU; plan runs so nothing needs both GPUs.

## 8. Policy training data and the switch to video

Policy training gets a `trajectory_source` setting with three modes:

| Mode | Object trajectory | Hands | When |
|---|---|---|---|
| `mocap` | motion capture | video | now, while video trajectories are poor |
| `mixed` | video where the clip's confidence is high, mocap elsewhere, labelled per clip | video | transition, to see whether video trajectories train as well |
| `video` | video only | video | the final pipeline |

The trainer refuses `mocap` and `mixed` when a `final` flag is set, and writes the mode into every checkpoint's
metadata. A policy trained in `mocap` mode is a development result, never a result reported for the pipeline.

**Gate for switching an object class to `video`** (starting values; revise once the benchmark has 8 or more
objects), measured on `test` clips against mocap:

- object position error: median at most 1.0 cm, 95th percentile at most 2.5 cm;
- rotation error: median at most 5 deg, 95th percentile at most 15 deg (modulo declared symmetry);
- no unresolved look-alike flip longer than 0.25 s;
- frame-0 error at most 1 cm and 3 deg;
- and the decisive check: a policy trained on the **video** reference and scored against **mocap** reaches at least
  95% of the score of a policy trained on the mocap reference, on the same clips.

For reference, the current best with scanned meshes on the challenge rig is 2.85 cm mean error, so the gate needs
roughly a threefold improvement. When a class passes, record the commit and numbers in `bench/history.csv` and
switch its default. Mocap recording continues after that only as a periodic check.

## 9. Order of work

1. **P0** benchmark, firewall, video-only baseline. Nothing else starts before this.
2. **P1.1-P1.3** capture fixes that need the lab (markers, calibration, sync), in parallel with step 3 because they
   need a person in the lab. Ask the user to schedule them.
3. **P3.1 + P3.4** symmetry-aware tracking and frame 0: cheapest large gain, code mostly exists.
4. **P2.2** corrected agent loop (depth-aware, surface gate), then the planner/builder comparison.
5. **P2.1** scan clips, then **P2.4-P2.5** symmetry files and the selector.
6. **P3.2-P3.3** refinement and physical constraints.
7. **P4** camera path from video, scored once camera markers exist.
8. **P5** hands.
9. **P3.5-P3.6** learned corrector and confidence, then the gate of section 8, class by class.

After each step: run the benchmark, add the row to `bench/history.csv`, write what changed and what it cost in
`NOTES.md`, and stop for review if a number moved the wrong way.

## 10. Working rules for the agent

- One change per commit, each with its benchmark row. Do not batch improvements: we need to know which one helped.
- Do not edit the pinned upstream code (`video_to_data`) or the challenge evaluator; wrap it.
- Small test first: one clip, one object, a few frames, before any full run.
- Report results as measured, including regressions and things not run. Say when a number comes from fewer than
  5 clips.
- Ask before deleting anything large, before any run over about 30 minutes, and before anything that needs `sudo`
  (the user runs those).
- Never handle API keys or tokens; the user enters them.
