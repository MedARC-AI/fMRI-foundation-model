#!/bin/bash

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/fMRI-foundation-model/fMRI-flat-MAE-ST"
cd $ROOT

# Set up python environment
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

PREFIXES=(
    output/vit_small_patch16_fmri_random_features
    output/pretrain_01/checkpoint-00009_features
    output/pretrain_01/checkpoint-00019_features
    output/pretrain_01/checkpoint-00029_features
    output/pretrain_01/checkpoint-00039_features
    output/pretrain_01/checkpoint-00049_features
    output/pretrain_01/checkpoint-00059_features
    output/pretrain_01/checkpoint-00069_features
    output/pretrain_01/checkpoint-00079_features
    output/pretrain_01/checkpoint-00089_features
    output/pretrain_01/checkpoint-00099_features
    output/connectome_features
)
OUTPATH="output/pretrain_01/task_linear_probe.json"

for prefix in ${PREFIXES[@]}; do
    python main_task_linear_probe.py \
        --feat_prefix $prefix \
        --output_path $OUTPATH
done
