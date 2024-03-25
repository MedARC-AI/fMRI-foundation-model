#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.11 -m venv fmri
source fmri/bin/activate

pip install numpy matplotlib jupyter jupyterlab_nvdashboard jupyterlab scipy tqdm scikit-learn scikit-image accelerate webdataset pandas matplotlib einops ftfy regex h5py torchvision torch==2.1.0 transformers xformers torchmetrics deepspeed wandb nilearn nibabel boto3 open_clip_torch kornia

pip install flash-attn --no-build-isolation
