#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=cpu32
#SBATCH --job-name=openneuro      
#SBATCH --time=300:00:00        # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err       # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc

source ~/.bashrc

cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation

jupyter nbconvert openneuro_minimal.ipynb --to python

srun python -u openneuro_minimal.py