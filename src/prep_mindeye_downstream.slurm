#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=arc
#SBATCH --job-name=ME_prepper
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=176     # 40 / 80 / 176 distributed across node
#SBATCH --time=06:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue

export model_name="NSDflat_large_gsrFalse_"
echo model_name=${model_name}
export latest_checkpoint="epoch99.pth"
echo latest_checkpoint=${latest_checkpoint}
export global_pool="False"
echo global_pool=${global_pool}

export NUM_GPUS=1 # Set to equal gres=gpu:#!
echo NUM_GPUS=$NUM_GPUS

source ~/.bashrc
cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/src/

jupyter nbconvert prep_mindeye_downstream.ipynb --to python
if [ $? -ne 0 ]; then
  echo "Error: Conversion of ipynb to Python failed. Exiting."
  exit 1
fi

python3 prep_mindeye_downstream.py ${model_name} ${latest_checkpoint}