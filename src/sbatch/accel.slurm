#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=p5
#SBATCH --job-name=found
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=176     # 40 / 80 / 176 distributed across node
#SBATCH --time=80:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue

export NUM_GPUS=2 # Set to equal gres=gpu:#!
echo NUM_GPUS=$NUM_GPUS

source ~/.bashrc
source /admin/home-paulscotti/foundation_env/bin/activate
cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/flat/src

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

export MODULEPATH=/opt/amazon/modules/modulefiles:$MODULEPATH

srun torchrun \
--nproc_per_node=$NUM_GPUS \
--nnodes=$SLURM_NNODES \
main.py