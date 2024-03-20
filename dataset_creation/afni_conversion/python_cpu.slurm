#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40
#SBATCH --job-name=openneuroMNI
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=4
#SBATCH --time=350:00:00        # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err       # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc

source ~/.bashrc

cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/afni_conversion

jupyter nbconvert MNI_normalization.ipynb --to python

export WORLD_SIZE=9#15

for ((i=0; i<$WORLD_SIZE; i++)); do
    export worker_id=$((i+1+15))
    echo worker_id=$worker_id
    python MNI_normalization.py $worker_id &
done

wait

# python MNI_normalization.py $SLURM_ARRAY_TASK_ID