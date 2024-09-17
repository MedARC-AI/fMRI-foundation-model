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

### 2. Save latents to hdf5 / parquet

- prep_mindeye_downstream.ipynb
- prep_HCP_downstream.ipynb

### 3. Evaluate downstream performanced using the saved latents

- mindeye_downstream.ipynb
- HCP_downstream.ipynb