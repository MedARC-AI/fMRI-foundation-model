#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.11 -m venv fmri
source fmri/bin/activate
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
# export PATH=/scratch/gpfs/qanguyen/mamba_fmri/fmri/lib/python3.11/site-packages/:$PATH
pip install numpy matplotlib jupyter jupyterlab_nvdashboard jupyterlab scipy ipykernel tqdm scikit-learn scikit-image accelerate webdataset pandas matplotlib einops ftfy regex h5py    transformers xformers torchmetrics deepspeed wandb nilearn nibabel boto3 open_clip_torch kornia huggingface_hub pytorch_lightning
