# Setting up environment
```bash
conda create --name openmmlab python=3.10 -y
conda activate openmmlab

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
conda install -c conda-forge gxx_linux-64  # For Linux
pip install imagesize p_tqdm

```

## Setup MMDetection
run `./setup_mmdet.sh` to setup mmdetection in the current conda environment

## Setup MMDeploy
run `./setup_mmdeploy.sh` to setup mmdetection in the current conda environment

> **Note:** The commit id used in the `setup_mmdet.sh` & `setup_mmdeploy.sh` is the one on which this repo is tested on. Later branches or commits might break somethings that can be fixed with minor changes
