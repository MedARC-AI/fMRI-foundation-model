#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=openneuroMNI
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=350:00:00        # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err       # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --array=0-5

source ~/.bashrc

cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/afni_conversion

jupyter nbconvert MNI_normalization.ipynb --to python

python MNI_normalization.py $SLURM_ARRAY_TASK_ID