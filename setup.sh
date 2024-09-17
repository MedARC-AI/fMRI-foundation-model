#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

# Ensure uv is installed for faster package installations
pip install uv

# Upgrade pip
pip install --upgrade pip

# Create a new virtual environment and activate it
python3.11 -m venv foundation_env
source foundation_env/bin/activate

# Use uv to install packages concurrently
uv pip install numpy matplotlib jupyter jupyterlab_nvdashboard jupyterlab ipywidgets scipy tqdm scikit-learn scikit-image accelerate webdataset pandas matplotlib einops ftfy regex h5py wandb nilearn nibabel boto3==1.34.57 open_clip_torch kornia omegaconf decord smart-open ffmpeg-python opencv-python==4.6.0.66 torchmetrics==1.3.0.post0 diffusers==0.23.0 pytorch-lightning==2.0.1 transformers==4.44.2 xformers==0.0.22.post7
uv pip install git+https://github.com/openai/CLIP.git --no-deps
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu120
uv pip install torchvision

# Install jupyter kernel spec
python -m ipykernel install --user --name=foundation_env