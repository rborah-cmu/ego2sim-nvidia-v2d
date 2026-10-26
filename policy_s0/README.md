# S0 per-episode policies (second Track 3 submission)

Policy training code for our second NVIDIA V2D Track 3 submission, written by our teammate on top of
the Ego2Sim reconstructions of the 20 evaluation episodes (object trajectories + meshes from this repo).
Checkpoints and data are not committed.

- `rian_eval/`: cleaning of the reconstructed object trajectories before training (`clean_refs.py`,
  `settle_spawn.py`, `*.sh`): removes false spin of the cups, fixes 180-degree flips of thin objects, and
  settles each object's starting pose in a free-fall simulation so it no longer starts inside the table.
- `s0/`: the S0 recipe (observation, action, reward, reset, EMA weights, VOC gate, features, train/eval scripts).
  One policy per episode, FlashSAC in FlashCHORD for 96.8M env steps.
  - Inputs: the future object trajectory, an object shape code, hand rays from the video, frozen image features.
    The reconstructed hand trajectory is not followed.
  - Rewards: object-keypoint tracking, plus a reward for both hands supporting the object.
  - Resets: practice resets just before each pickup, plus slightly displaced object starts.
  - Checkpoint per episode: the best by AUC over 64 parallel rollouts against the cleaned reference
    (weights are an exponential moving average).
- `cfg/`: Hydra configs (`observation/policy_s0.yaml`, `action/relative_hand_pose.yaml`,
  `objective/tracking_s0.yaml`, `reset/s0_pickup.yaml`, `curriculum/tracking_96m_pickup_ray5decay.yaml`).
- `train_logged.py`: runs FlashCHORD's `train_flash_sac.py` with W&B replaced by a local JSONL logger
  (FlashCHORD source untouched). `monitor_training.py`: training monitor.

Rollouts were produced by the official kit evaluator (pinned `video_to_data@8c778f9`), 1 world per episode,
and packed with the kit's `tools/pack_submission.py`.
