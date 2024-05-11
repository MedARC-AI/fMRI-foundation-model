#!/bin/bash

#SBATCH --job-name=pretrain_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --gpus-per-node=v100-32:4
#SBATCH --time=2-00:00:00
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/fMRI-foundation-model/fMRI-flat-MAE-ST"
cd $ROOT

# Set up python environment
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

torchrun --standalone --nproc_per_node=4 \
    main_pretrain.py \
    --output_dir output \
    --name pretrain_01 \
    --model mae_vit_small_patch16_fmri \
    --mask_ratio 0.75 \
    --decoder_depth 4 \
    --num_frames 16 \
    --t_patch_size 2 \
    --pred_t_dim 8 \
    --epochs 100 \
    --warmup_epochs 5 \
    --batch_size 32 \
    --blr 1.0e-3 \
    --clip_grad 1.0 \
    --wandb \
    --distributed
