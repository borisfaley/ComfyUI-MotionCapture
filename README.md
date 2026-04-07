# ComfyUI-MotionCapture

Форк [PozzettiAndrea/ComfyUI-MotionCapture](https://github.com/PozzettiAndrea/ComfyUI-MotionCapture), адаптированный для Python 3.13 + PyTorch 2.10 + CUDA 12.8.

## Изменения относительно оригинала

- Убрана зависимость от `comfy_env`. Ноды импортируются напрямую, без слоя изоляции.
- CUDA-расширения DPVO (`cuda_corr`, `cuda_ba`, `lietorch_backends`) пропатчены для компиляции с PyTorch 2.10 — заменён устаревший `.type()` на `.scalar_type()` в CUDA-ядрах, обновлён макрос в `dispatch.h`.
- `bpy` обновлён с 4.2 до 5.1 (единственная доступная версия для Python 3.13).
- `torch_scatter` установлен с PyG wheel index для cu128.
- Все Python-зависимости ставятся напрямую в conda-окружение вместо изолированного venv через comfy-env.

## Установка (Linux)

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

## Установка (Windows 11)

Требуется: Python 3.13, CUDA Toolkit 12.8, Visual Studio 2022 (компонент "Разработка классических приложений на C++"), Git.

### 1. Conda-окружение и PyTorch

```powershell
conda create -n comfy python=3.13 -y
conda activate comfy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Python-зависимости

```powershell
pip install smplx scipy lightning hydra-core hydra-zen hydra-colorlog ^
  timm einops rich termcolor joblib scikit-image imageio ^
  ffmpeg-python trimesh chumpy-fork pycolmap bvh lapx ^
  roma braceexpand webdataset optree pyrootutils ^
  comfy_aimdo pypose bpy
```

Примечание: `pyrender` на Windows может не установиться из-за зависимости от `pyglet`/OpenGL. Если установка падает — пропустите, визуализация SMPL будет работать через встроенные viewer-ноды.

### 3. torch_scatter

```powershell
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

### 4. DPVO (опционально, для точной одометрии камеры)

Сборка CUDA-расширений на Windows требует Visual Studio 2022 и CUDA Toolkit.

```powershell
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO

# Скачать Eigen
curl -L -o eigen-3.4.0.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
tar -xf eigen-3.4.0.zip -C thirdparty
```

Перед сборкой нужно пропатчить файлы (заменить `.type()` на `.scalar_type()`):

- `dpvo/altcorr/correlation_kernel.cu` — 4 замены (`fmap1.type()` и `net.type()`)
- `dpvo/lietorch/src/lietorch_gpu.cu` — все `a.type()` и `X.type()`
- `dpvo/lietorch/src/lietorch_cpu.cpp` — все `a.type()` и `X.type()`
- `dpvo/lietorch/include/dispatch.h` — убрать `::detail::scalar_type(the_type)`, заменить на просто `TYPE`

Затем собрать из Developer Command Prompt for VS 2022:

```powershell
set DISTUTILS_USE_SDK=1
pip install . --no-build-isolation
```

Если DPVO не собирается — не критично. Нода будет работать с SimpleVO вместо DPVO.

### 5. Установка ноды

Склонировать в папку `custom_nodes` вашего ComfyUI:

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/borisfaley/ComfyUI-MotionCapture.git
```

## Протестировано

- Rocky Linux 9, Python 3.13, PyTorch 2.10+cu128, NVIDIA RTX 4090 — 22 ноды, DPVO доступен
- Windows 11 — установка зависимостей совместима, DPVO требует сборки из исходников
