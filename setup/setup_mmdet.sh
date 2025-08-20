# Cloning MMDetection
conda activate openmmlab

cd ../third_party/

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git config --global --add safe.directory "$(pwd)"
git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d
pip install -v -e .


pip install p_tqdm future tensorboard omnixai
