#!/bin/bash

#SBATCH --job-name=extract_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --gpus-per-node=v100-32:1
#SBATCH --time=01:00:00
#SBATCH --account=med230001p
#SBATCH --array=0-11

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/fMRI-foundation-model/fMRI-flat-MAE-ST"
cd $ROOT

# Set up python environment
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

SPLIT="train"
SAMPLES=100000

MODELS=(
    "connectome"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
    "vit_small_patch16_fmri"
)

CKPTS=(
    ""
    ""
    "output/pretrain_01/checkpoint-00009.pth"
    "output/pretrain_01/checkpoint-00019.pth"
    "output/pretrain_01/checkpoint-00029.pth"
    "output/pretrain_01/checkpoint-00039.pth"
    "output/pretrain_01/checkpoint-00049.pth"
    "output/pretrain_01/checkpoint-00059.pth"
    "output/pretrain_01/checkpoint-00069.pth"
    "output/pretrain_01/checkpoint-00079.pth"
    "output/pretrain_01/checkpoint-00089.pth"
    "output/pretrain_01/checkpoint-00099.pth"
)

OUTPATHS=(
    "output/connectome_features_${SPLIT}.parquet"
    "output/vit_small_patch16_fmri_random_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00009_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00019_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00029_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00039_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00049_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00059_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00069_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00079_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00089_features_${SPLIT}.parquet"
    "output/pretrain_01/checkpoint-00099_features_${SPLIT}.parquet"
)

MODEL=${MODELS[SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[SLURM_ARRAY_TASK_ID]}
OUTPATH=${OUTPATHS[SLURM_ARRAY_TASK_ID]}

python main_extract.py \
    --output_path $OUTPATH \
    --model "${MODEL}" \
    --ckpt_path "${CKPT}" \
    --split $SPLIT \
    --num_samples $SAMPLES \
    --batch_size 32
