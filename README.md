# ComfyUI-MotionCapture

Форк [PozzettiAndrea/ComfyUI-MotionCapture](https://github.com/PozzettiAndrea/ComfyUI-MotionCapture), адаптированный для Python 3.13 + PyTorch 2.10 + CUDA 12.8 (Rocky Linux).

## Изменения относительно оригинала

- Убрана зависимость от `comfy_env`. Ноды импортируются напрямую, без слоя изоляции.
- CUDA-расширения DPVO (`cuda_corr`, `cuda_ba`, `lietorch_backends`) пропатчены для компиляции с PyTorch 2.10 — заменён устаревший `.type()` на `.scalar_type()` в CUDA-ядрах, обновлён макрос в `dispatch.h`.
- `bpy` обновлён с 4.2 до 5.1 (единственная доступная версия для Python 3.13).
- `torch_scatter` установлен с PyG wheel index для cu128.
- Все Python-зависимости ставятся напрямую в conda-окружение вместо изолированного venv через comfy-env.

## Установка

Требуется conda-окружение с Python 3.13, PyTorch 2.10+cu128.

```bash
# Python-зависимости
pip install smplx scipy lightning hydra-core hydra-zen hydra-colorlog \
  timm einops rich termcolor joblib scikit-image imageio \
  ffmpeg-python trimesh chumpy-fork pycolmap bvh lapx \
  roma braceexpand webdataset optree pyrootutils \
  comfy_aimdo pyrender pypose bpy

# torch_scatter
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# CUDA-расширения DPVO (сборка из исходников)
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
# Патч .type() -> .scalar_type() в файлах:
#   dpvo/altcorr/correlation_kernel.cu
#   dpvo/lietorch/src/lietorch_gpu.cu
#   dpvo/lietorch/src/lietorch_cpu.cpp
#   dpvo/lietorch/include/dispatch.h
pip install . --no-build-isolation
```

## Протестировано

- Rocky Linux 9, Python 3.13, PyTorch 2.10+cu128, NVIDIA RTX 4090
- 22 ноды загружаются, DPVO доступен
