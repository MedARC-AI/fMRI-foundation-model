#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=p5
#SBATCH --job-name=finetuneHCP
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --time=12:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --qos=idle

export NUM_GPUS=1 # Set to equal gres=gpu:#!
echo NUM_GPUS=$NUM_GPUS

source ~/.bashrc
source /admin/home-ckadirt/foundation_env/bin/activate
cd /weka/proj-fmri/ckadirt/fMRI-foundation-model/src

jupyter nbconvert HCP_downstream_finetune.ipynb --to python
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

export global_pool=True
python HCP_downstream_finetune.py HCPflat_large_gsrFalse_ epoch99.pth