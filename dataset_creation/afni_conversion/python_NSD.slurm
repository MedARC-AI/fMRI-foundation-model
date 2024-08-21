#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=nsd1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=100:00:00        # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err       # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc

source ~/.bashrc

cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/afni_conversion

jupyter nbconvert MNI_normalization_NSD.ipynb --to python

python MNI_normalization_NSD.py