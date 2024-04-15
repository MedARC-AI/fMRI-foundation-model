#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.10 -m venv found
source found/bin/activate

pip install numpy matplotlib jupyter jupyterlab_nvdashboard jupyterlab scipy tqdm scikit-learn scikit-image accelerate webdataset pandas matplotlib einops ftfy regex h5py transformers xformers==0.0.22.post7 torchmetrics==1.3.1 wandb nilearn nibabel boto3==1.34.57 open_clip_torch kornia omegaconf decord smart-open ffmpeg-python opencv-python==4.6.0.66

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pip install torch==2.1.0 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121 --no-deps

# If you are using Stability HPC then do this (may need to fix the absolute path to .whl):
# pip install fMRI-foundation-model/torchdata_whl/torchdata-0.5.1+cb9ed24-cp310-cp310-linux_x86_64.whl
# pip install s3fs==2024.2.0

# If you are NOT using Stability HPC then do this:
pip install torchdata==0.6.1 --no-deps
