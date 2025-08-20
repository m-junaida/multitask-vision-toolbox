
# Cloning MMPretrain
conda activate openmmlab

cd ../third_party/

git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
git config --global --add safe.directory "$(pwd)"
git checkout ee7f2e88501f61aa95c742dd5f429f039935ee90
pip install -v -e .
