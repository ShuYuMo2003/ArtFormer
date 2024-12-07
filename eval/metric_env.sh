conda create -n pytorch3d python=3.9 -y
source activate pytorch3d
which pip
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
pip install "numpy<2"
pip install tqdm
pip install trimesh
pip install imageio
pip install trimesh
pip install point_cloud_utils
pip install rich