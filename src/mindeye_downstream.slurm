#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=arc
#SBATCH --job-name=ME_downstream
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1       # Set to equal gres=gpu:#!
#SBATCH --cpus-per-task=176     # 40 / 80 / 176 distributed across node
#SBATCH --time=50:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue

export parquet_path="/weka/proj-fmri/paulscotti/fMRI-foundation-model/flat/checkpoints/NSDflat_large_gsrFalse__gpFalse/epoch99/test.parquet"
echo parquet_path=${parquet_path}

export NUM_GPUS=1 # Set to equal gres=gpu:#!
echo NUM_GPUS=$NUM_GPUS
export BATCH_SIZE=24 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

source ~/.bashrc
cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/src/

jupyter nbconvert MindTrain.ipynb --to python
if [ $? -ne 0 ]; then
  echo "Error: Conversion of ipynb to Python failed. Exiting."
  exit 1
fi

python3 MindTrain.py --data_path=/weka/proj-medarc/shared/mindeyev2_dataset --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=50 --no-use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --no-ckpt_saving --wandb_log