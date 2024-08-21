#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.12 -m venv found
source found/bin/activate

pip3 install torch torchvision torchaudio

pip install numpy matplotlib jupyter jupyterlab_nvdashboard jupyterlab scipy tqdm scikit-learn scikit-image accelerate webdataset pandas matplotlib einops ftfy regex h5py transformers wandb nilearn nibabel open_clip_torch kornia omegaconf decord smart-open ffmpeg-python simplejson iopath
