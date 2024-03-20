#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=mae
#SBATCH --nodes=2              
#SBATCH --gres=gpu:8
#SBATCH --time=20:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8

export NUM_GPUS=8  # Set to equal gres=gpu:#!
export NUM_NODES=2

source /weka/proj-fmri/souvik/foundational_model/found/bin/activate
cd /weka/proj-fmri/souvik/foundational_model/fMRI-MAE/
jupyter nbconvert main.ipynb --to python

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

srun python main.py
