# Cloning MMDeploy
conda activate openmmlab

cd ../third_party/

git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git config --global --add safe.directory "$(pwd)"
git checkout 3f8604bd72e8e15d06b2e0552fe2fdb8f8de33c4
pip install -v -e .
