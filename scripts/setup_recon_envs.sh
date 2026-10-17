#!/usr/bin/env bash
# Native replacements for the V2D reconstruction Docker images on riddle (no Docker).
# Each image <name> becomes a uv venv at ~/v2d/envs/<name>/.venv that mirrors its Dockerfile;
# ~/v2d/bin/docker (docker_shim.py) runs `python -m MODULE` inside it.
#
# Usage: setup_recon_envs.sh host|v2d_moge|v2d_sam2|v2d_foundation_pose|v2d_task_library_loader|rg_retarget|v2d_wilor|v2d_hamer ...
set -euo pipefail
source ~/v2d/env.sh
REPO=~/v2d/vtd_pinned
M=$REPO/reconstruction/modules
ENVS=~/v2d/envs
# Dockerfile base: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel (Python 3.11).
TORCH_CU124=(torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124)

new_env() {  # new_env NAME PYVER
  # uv-managed Python only: riddle's /usr/bin/python3.11 has no headers (Python.h), which breaks CUDA/C++ builds.
  mkdir -p "$ENVS/$1"
  uv venv -q --seed --clear --python-preference only-managed --python "$2" "$ENVS/$1/.venv"
}
pip_in() {  # pip_in NAME args...
  local name=$1; shift
  VIRTUAL_ENV="$ENVS/$name/.venv" uv pip install --python "$ENVS/$name/.venv/bin/python" "$@"
}

setup_host() {
  # Host-side orchestration packages (install_ego_reconstruction_packages.sh), plus pyarrow for our tools.
  new_env recon_host 3.11
  (cd "$REPO/reconstruction" && pip_in recon_host Pillow "scipy>=1.13,<2" pyarrow \
    -e modules/v2d_common -e modules/v2d_docker -e modules/v2d_depth \
    -e modules/v2d_anycalib/docker -e modules/v2d_geocalib/docker -e modules/v2d_droid_slam/docker \
    -e modules/v2d_ego_hand_reconstruction/docker -e modules/v2d_foundation_pose/docker \
    -e modules/v2d_grounding_dino/docker -e modules/v2d_gsplat_refinement/docker -e modules/v2d_hamer/docker \
    -e modules/v2d_hand_alignment/docker -e modules/v2d_hawor/docker -e modules/v2d_mediapipe/docker \
    -e modules/v2d_moge/docker -e modules/v2d_sam2/docker -e modules/v2d_sam3d/docker -e modules/v2d_wilor/docker \
    -e modules/v2d_pipelines)
}

setup_v2d_moge() {
  # Dockerfile: pip install -e v2d_common -e v2d_moge/lib
  new_env v2d_moge 3.11
  pip_in v2d_moge "${TORCH_CU124[@]}"
  pip_in v2d_moge "huggingface_hub[cli]" -e "$M/v2d_common" -e "$M/v2d_moge/lib"
}

setup_v2d_sam2() {
  # Dockerfile: pip install -e v2d_common[io] -e v2d_mv -e v2d_sam2/lib
  # No nvcc on riddle -> skip SAM2's optional CUDA extension (only used for mask hole filling).
  new_env v2d_sam2 3.11
  pip_in v2d_sam2 "${TORCH_CU124[@]}"
  pip_in v2d_sam2 setuptools wheel
  SAM2_BUILD_CUDA=0 pip_in v2d_sam2 --no-build-isolation "SAM-2 @ git+https://github.com/facebookresearch/sam2.git"
  pip_in v2d_sam2 "huggingface_hub[cli]" -e "$M/v2d_common[io]" -e "$M/v2d_mv" -e "$M/v2d_sam2/lib"
}

use_cuda124() {
  # CUDA 12.4.1 toolkit (nvcc) installed with micromamba into ~/v2d/cuda/12.4 (matches the images' cu124 torch).
  export CUDA_HOME=~/v2d/cuda/12.4 CUDA_PATH=~/v2d/cuda/12.4
  export PATH="$CUDA_HOME/bin:$PATH"
  # riddle has only sm_89 GPUs; the images also build 8.0/8.6/9.0 (we skip them to halve build time).
  export TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS:-12}
}

setup_v2d_foundation_pose() {
  # Mirrors v2d_foundation_pose/docker/Dockerfile. apt deps that matter (Eigen3, Boost system/program_options,
  # OpenMP, EGL/GL, cmake, gcc 13.3) are already on riddle; pybind11 comes from pip instead of a source build.
  use_cuda124
  local E=v2d_foundation_pose PY=$ENVS/v2d_foundation_pose/.venv/bin/python
  local FP=$M/v2d_foundation_pose/lib/FoundationPose
  new_env $E 3.11
  pip_in $E "${TORCH_CU124[@]}"
  pip_in $E setuptools wheel ninja "pybind11==2.10.0"
  pip_in $E scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations \
    "warp-lang==1.14.0" einops kornia pyrender
  pip_in $E kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
  pip_in $E --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
  pip_in $E --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
  # warp-lang pinned: latest (1.18) needs a CUDA 13 driver; 1.14.0 is the CUDA 12 build FlashCHORD also uses.
  # tensorrt-cu12 alone installs without its bindings under uv -> name bindings/libs explicitly.
  pip_in $E scikit-image meshcat webdataset omegaconf pypng roma seaborn opencv-contrib-python openpyxl wandb \
    h5py fast_simplification gdown "warp-lang==1.14.0" "tensorrt-cu12==10.7.0.post1" \
    "tensorrt-cu12-bindings==10.7.0.post1" "tensorrt-cu12-libs==10.7.0.post1"
  # mycpp (pybind11 + Eigen + Boost + OpenMP), built in-tree as in the image.
  (cd "$FP/mycpp" && mkdir -p build && cd build \
    && cmake .. -DPYTHON_EXECUTABLE="$PY" -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" >/dev/null \
    && make -j"$MAX_JOBS")
  # bundlesdf/mycuda CUDA extensions (common, gridencoder).
  (cd "$FP/bundlesdf/mycuda" && pip_in $E --no-build-isolation -e .)
  pip_in $E -e "$M/v2d_common[io]" -e "$M/v2d_mv[math]" -e "$M/v2d_mesh/lib" -e "$M/v2d_foundation_pose/lib"
}

setup_v2d_task_library_loader() {
  # Mirrors v2d_task_library_loader/docker/Dockerfile (pytorch 2.8.0-cuda12.8 base). pytorch3d is optional there
  # (trimesh fallback in distance_utils) -> skipped. robotic_grounding comes from the pinned monorepo via a .pth,
  # instead of the image's staged copy (_rg_pkg).
  local E=v2d_task_library_loader
  new_env $E 3.11
  pip_in $E torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip_in $E setuptools wheel numpy opencv-python-headless deprecation
  pip_in $E --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy" \
    "manotorch @ git+https://github.com/lixiny/manotorch"
  pip_in $E -e "$M/v2d_common" -e "$M/v2d_task_library_loader/lib"
  local SP
  SP=$("$ENVS/$E/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$REPO/robotic_grounding/source/robotic_grounding" > "$SP/robotic_grounding_src.pth"
}

setup_rg_retarget() {
  # robotic_grounding retarget stage (ego_recon_to_sharpa.py, generate_rigid_urdfs.py, reconstruct_support_surfaces.py)
  # without Isaac Lab: the pip pins of robotic_grounding/workflow/Dockerfile that these scripts import. Skipped (optional,
  # lazily imported): py-soma-x, judo-rai, onnxruntime, wandb, datasets. numpy stays 2.x: cmeel-boost needs >=2.2 (the
  # image force-downgrades to <2 afterwards for Isaac Sim, which we do not use).
  local E=rg_retarget
  new_env $E 3.11
  pip_in $E torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip_in $E setuptools wheel numpy scipy scikit-learn trimesh pyarrow tqdm psutil plotly "viser==1.0.15" \
    imageio imageio-ffmpeg usd-core deprecation loop-rate-limiters mujoco \
    "pin==3.7.0" "pin-pink==4.2.0" "coal==3.0.1" "eigenpy==3.10.3" "cmeel==0.57.3" "cmeel-boost==1.87.0.1" \
    "cmeel-urdfdom==4.0.1" "cmeel-console-bridge==1.0.2.3" "cmeel-tinyxml2==10.0.0" "cmeel-tinyxml==2.6.2.3" \
    "cmeel-assimp==5.4.3.1" "cmeel-octomap==1.10.0" "cmeel-qhull==8.0.2.1" "cmeel-zlib==1.3.1" \
    osqp "qpsolvers[clarabel,daqp,proxqp,scs,osqp]==4.9.0"
  pip_in $E toml  # setup.py reads its extension.toml
  pip_in $E --no-build-isolation -e "$REPO/robotic_grounding/source/robotic_grounding"
}

egl_pth() {  # the images set ENV PYOPENGL_PLATFORM=egl (headless pyrender); the shim ignores Dockerfile ENV -> .pth
  local SP
  SP=$("$ENVS/$1/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo 'import os; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")' > "$SP/zz_pyopengl_egl.pth"
}

setup_v2d_wilor() {
  # Mirrors v2d_wilor/docker/Dockerfile (pytorch 2.5.1-cuda12.4 base; WiLoR-mini from git).
  local E=v2d_wilor
  new_env $E 3.11
  pip_in $E "${TORCH_CU124[@]}"
  pip_in $E setuptools wheel "numpy<2" Pillow opencv-python tqdm pyrender trimesh smplx yacs hydra-core omegaconf \
    einops timm huggingface_hub transformers safetensors ultralytics dill
  pip_in $E --no-build-isolation "manotorch @ git+https://github.com/lixiny/manotorch.git" \
    "chumpy @ git+https://github.com/mattloper/chumpy.git"
  pip_in $E "wilor-mini @ git+https://github.com/warmshao/WiLoR-mini.git"
  pip_in $E -e "$M/v2d_common" -e "$M/v2d_wilor/lib"
  egl_pth $E
  "$ENVS/$E/.venv/bin/python" -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('wilor_mini import OK')"
}

setup_v2d_hamer() {
  # Mirrors v2d_hamer/docker/Dockerfile: HaMeR clone at the image's pinned commit (no-deps editable, + .pth like its
  # ENV PYTHONPATH=/opt/hamer). Also provides the depth alignment step (run_align_hands).
  local E=v2d_hamer SRC=~/v2d/src/hamer
  if [ ! -d "$SRC/.git" ]; then
    git clone -q --recursive https://github.com/geopavlakos/hamer.git "$SRC"
  fi
  (cd "$SRC" && git checkout -q 3a01849f4148352e9260b69bf28b65d1671a4905 && git submodule update -q --init --recursive)
  new_env $E 3.11
  pip_in $E "${TORCH_CU124[@]}"
  pip_in $E setuptools wheel "numpy<2" Pillow opencv-python tqdm pyrender trimesh smplx yacs hydra-core omegaconf \
    pytorch-lightning timm einops webdataset braceexpand dill av joblib loguru scikit-image
  pip_in $E --no-build-isolation "manotorch @ git+https://github.com/lixiny/manotorch.git" \
    "chumpy @ git+https://github.com/mattloper/chumpy.git"
  pip_in $E --no-deps -e "$SRC"
  pip_in $E -e "$M/v2d_common" -e "$M/v2d_hamer/lib"
  local SP
  SP=$("$ENVS/$E/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$SRC" > "$SP/hamer_src.pth"
  egl_pth $E
  "$ENVS/$E/.venv/bin/python" -c "from hamer.models import load_hamer; from hamer.datasets.utils import generate_image_patch_cv2; print('hamer imports OK')"
}

for target in "$@"; do
  echo "=== $target"; SECONDS=0
  "setup_${target}"
  echo "=== $target done in ${SECONDS}s; $(du -sh "$ENVS/$target" | cut -f1)"
done
