# ComfyUI-MotionCapture

Fork of [PozzettiAndrea/ComfyUI-MotionCapture](https://github.com/PozzettiAndrea/ComfyUI-MotionCapture), adapted to run on Python 3.13 + PyTorch 2.10 + CUDA 12.8 (Rocky Linux).

## Changes from upstream

- Removed `comfy_env` dependency. Nodes are imported directly, no isolation layer.
- DPVO CUDA extensions (`cuda_corr`, `cuda_ba`, `lietorch_backends`) patched to compile against PyTorch 2.10 (replaced deprecated `.type()` with `.scalar_type()` in CUDA kernels, updated `dispatch.h` macro).
- `bpy` updated from 4.2 to 5.1 (only version available for Python 3.13).
- `torch_scatter` installed from PyG wheel index for cu128.
- All Python dependencies installed directly into the conda environment instead of comfy-env isolated venv.

## Installation

Requires conda environment with Python 3.13, PyTorch 2.10+cu128.

```bash
# Python dependencies
pip install smplx scipy lightning hydra-core hydra-zen hydra-colorlog \
  timm einops rich termcolor joblib scikit-image imageio \
  ffmpeg-python trimesh chumpy-fork pycolmap bvh lapx \
  roma braceexpand webdataset optree pyrootutils \
  comfy_aimdo pyrender pypose bpy

# torch_scatter
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# DPVO CUDA extensions (build from source)
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
# Patch .type() -> .scalar_type() in:
#   dpvo/altcorr/correlation_kernel.cu
#   dpvo/lietorch/src/lietorch_gpu.cu
#   dpvo/lietorch/src/lietorch_cpu.cpp
#   dpvo/lietorch/include/dispatch.h
pip install . --no-build-isolation
```

## Tested on

- Rocky Linux 9, Python 3.13, PyTorch 2.10+cu128, NVIDIA RTX 4090
- 22 nodes load, DPVO available
