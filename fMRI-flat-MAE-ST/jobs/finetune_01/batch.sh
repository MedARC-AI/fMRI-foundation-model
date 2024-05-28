#!/bin/bash

#SBATCH --job-name=finetune_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --gpus-per-node=v100-32:1
#SBATCH --time=2-00:00:00
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/fMRI-foundation-model/fMRI-flat-MAE-ST"
cd $ROOT

# Set up python environment
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

torchrun --standalone --nproc_per_node=1 \
    main_finetune.py \
    --output_dir output \
    --name finetune_01 \
    --model vit_small_patch16_fmri \
    --num_frames 16 \
    --t_patch_size 2 \
    --global_pool avg \
    --target trial_type \
    --finetune output/pretrain_01/checkpoint-00099.pth \
    --freeze_params '*' \
    --unfreeze_params 'blocks.*.attn.*,norm.*,spatial_pool.*,head.*' \
    --epochs 30 \
    --warmup_epochs 2 \
    --batch_size 32 \
    --num_train_samples 100000 \
    --num_val_samples 5000 \
    --blr 5e-3 \
    --clip_grad 5.0 \
    --drop_path_rate 0.1 \
    --dropout 0.3 \
    --layer_decay 0.65 \
    --wandb
