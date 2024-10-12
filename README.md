# fMRI Foundation Model

In-progress -- this repo is under active development in the MedARC discord server. https://medarc.ai/fmri

## Installation

- Run setup.sh to create a new "foundation_env" virtual environment

- Activate the virtual environment with "source foundation_env/bin/activate"

## Datasets

- https://huggingface.co/datasets/bold-ai/HCP-Flat
- https://huggingface.co/datasets/bold-ai/NSD-Flat

## Usage

### 1. Train MAE

- main.ipynb (use accel.slurm to allocate multi-gpu Slurm job)

### 2a. Downstream probe using frozen MAE latents

Save latents to hdf5 / parquet:

- prep_mindeye_downstream.ipynb
- prep_HCP_downstream.ipynb

Then evaluate downstream performance using the saved latents:

- mindeye_downstream.ipynb
- HCP_downstream.ipynb

### 2b. Full fine-tuning of both MAE and downstream model

This requires having access to train_subj01.hdf5 which is saved in "/weka/proj-fmri/paulscotti/fMRI-foundation-model/src".

If you cannot access this file, the commented out code shows how to create this file yourself.

- mindeye_finetuning.ipynb