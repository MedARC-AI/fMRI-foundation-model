#!/bin/bash
#SBATCH --partition=della_gpu
#SBATCH --job-name=downstream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00         # total run time limit (HH:MM:SS)
#SBATCH -o slurms/%j.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=20G

# srun --pty -p della-gpu  -t 4:00:00 --cpus-per-task=24 --ntasks-per-node=1 --gres=gpu:1 --mem-per-cpu=20G bash
source ~/.bashrc
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
cd /scratch/gpfs/qanguyen/mamba_fmri/fMRI-MAE
module load cudatoolkit/11.7
# source ../fmri/bin/activate
conda activate /scratch/gpfs/qanguyen/mamba_fmri/fmri

jupyter nbconvert downstream.ipynb --to python
# module load openmpi/cuda-11.1/gcc/4.1.1 cudatoolkit/11.1
export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=32
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}
echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
echo model_name=${model_name}
# accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 downstream.py
python -u downstream.py