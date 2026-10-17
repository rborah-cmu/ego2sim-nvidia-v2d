# ego2sim-nvidia-v2d
Nvidia Video2Data Challenge using Ego2Sim

# Ego2Sim — V2D Challenge Track 3 pipeline (ego video -> object meshes + tracks + hands -> floating Sharpa policy)

Code that produced our Track 3 submission, run against [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data) at the pinned evaluator commit `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef` (upstream files unchanged; this repo holds only our code; the official
evaluator `robotic_grounding/flash_chord/scripts/evaluate_policy.py` is used as is).

Per evaluation episode (`scripts/ego2sim_public_episode.sh EP OBJ1,OBJ2 GPU`, SPLIT=evaluation, MESH_SRC=meshes dir):
1. `tools/ego_stereo_depth.py` — undistorted cam_a + SGBM stereo depth from the two ego cameras (metric scale).
2. SAM2 masks: object prompts `tools/obj_prompts.py` (boxes/points read off frame sheets `tools/make_grid.py`),
   hand prompts from WiLoR detections `tools/auto_hand_prompts.py`.
3. MoGe depth fused with stereo `tools/fuse_depth.py`; camera motion from stereo VO `tools/ego_vo.py`.
4. Hands: WiLoR + depth alignment `tools/ego_hands.py`, glitch repair `tools/clean_hand_tracks.py`,
   occluded-grasp gap filling carried with the object `tools/fill_hand_gaps_object.py`.
5. Object meshes: AgentSTAR Ego2Sim shape-only harness (`agentstar_harness/`, captures built by
   `scripts/e2s_make_capture.sh`, exported by `tools/agentstar_mesh_export.py`); 6-DoF tracks with FoundationPose.
6. Bundle (`tools/build_ego_bundle.py`) -> task-library loader -> retarget to floating Sharpa -> table support
   (`scripts/ego2sim_episode.sh`, `tools/make_table_support.py`) -> multi-object partition (`tools/merge_sharpa_objects.py`).
7. FlashSAC training (`scripts/train_episode.sh`, `scripts/e2s_train_queue_v2.sh`: 40.27M steps, compressed VOC
   curriculum; 2x budget retry when MPPE vs our reference > 3 cm), official evaluator on every checkpoint
   (`scripts/eval_all_checkpoints.sh`), best checkpoint by MPPE vs our own reference (no ground truth used).
8. `scripts/pack_eval.sh` -> kit `tools/pack_submission.py`.

Orchestration on 2x RTX 4090: `scripts/eval_pre_queue.sh`, `scripts/eval_worker.sh`, `scripts/e2s_mesh_watch.sh`.
Validation against public mocap: `tools/score_public_episode.py` (kit metric code), `tools/diagnose_rollout.py`.
MANO model files are NOT included (licence); paths assume a `~/v2d` layout.