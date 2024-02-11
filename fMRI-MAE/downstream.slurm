#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=downstream
#SBATCH --nodes=1              
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00         # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --exclusive

source ~/.bashrc
cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/fMRI-MAE/
jupyter nbconvert downstream.ipynb --to python

export NUM_GPUS=4  # Set to equal gres=gpu:#!
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

model_name="mae_testing"
mae_model_name="mae_testing"
echo model_name=${model_name}
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 downstream.py \
--model_name=${model_name} \
--mae_model_name=${mae_model_name} \
--nsd_wds_path=/weka/proj-fmri/shared/mindeyev2_dataset/wds \
--nsd_raw_path=/weka/proj-fmri/paulscotti/fMRI-foundation-model/nsd \
--nsd_image_path=/weka/proj-fmri/shared/mindeyev2_dataset \
--batch_size=${BATCH_SIZE} \
--num_epoch=12 --num_sessions=10 \
--in_dim=410112 --hidden_dim=512 \
--use_contrastive_loss \
--tube_mask_ratio=0.75 --decoder_mask_ratio=0.75 \
--num_frames=4 --tubelet_size=1 \
--patch_size=8 --frame_patch_size=1 \
--max_lr=3e-5 --seed=42 --wandb_log --ckpt_saving