#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=mae
#SBATCH --nodes=1              
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --exclusive


source ~/.bashrc

cd /weka/home-alexnguyen/mamba_fmri/fMRI-MAE/
source /weka/home-alexnguyen/mamba_fmri/fmri/bin/activate
jupyter nbconvert main.ipynb --to python

# Make sure another job doesnt use same port, here using random number
export NUM_GPUS=4  # Set to equal gres=gpu:#!

export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 \
    main.py
# srun python main.py