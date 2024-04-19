#!/bin/bash
#SBATCH --partition=della_gpu
#SBATCH --job-name=fmrimamba
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --time=50:00:00         # total run time limit (HH:MM:SS)
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
# SBATCH --no-requeue
# SBATCH --exclusive

export NUM_GPUS=1 # Set to equal gres=gpu:#!
echo NUM_GPUS=$NUM_GPUS
cd /scratch/gpfs/qanguyen/mamba_fmri/fMRI-MAE
conda activate /scratch/gpfs/qanguyen/mamba_fmri/fmri

jupyter nbconvert main.ipynb --to python
if [ $? -ne 0 ]; then
  echo "Error: Conversion of ipynb to Python failed. Exiting."
  exit 1
fi

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=$((${SLURM_NNODES} * ${NUM_GPUS}))

# Other settings relevant for multi-node
export NCCL_DEBUG=WARN #INFO
export NCCL_PROTO=simple
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=1
# module load openmpi cuda/12.1
wandb offline
srun torchrun \
--nproc_per_node=$NUM_GPUS \
--nnodes=$SLURM_NNODES \
--rdzv_id=$SLURM_JOBID \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:$MASTER_PORT \
--rdzv_conf=timeout=90 \
main.py

if [ $? -ne 0 ]; then
  echo "Error: srun command failed. Please check the logs for more details."
  exit 1
fi