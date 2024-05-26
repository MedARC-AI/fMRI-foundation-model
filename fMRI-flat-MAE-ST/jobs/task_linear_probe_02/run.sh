#!/bin/bash

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/fMRI-foundation-model/fMRI-flat-MAE-ST"
cd $ROOT

# Set up python environment
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

TARGET="trial_type"
PREFIXES=(
    # output/vit_small_patch16_fmri_random_features_event
    # output/pretrain_01/checkpoint-00009_features_event
    # output/pretrain_01/checkpoint-00019_features_event
    # output/pretrain_01/checkpoint-00029_features_event
    # output/pretrain_01/checkpoint-00039_features_event
    # output/pretrain_01/checkpoint-00049_features_event
    # output/pretrain_01/checkpoint-00059_features_event
    # output/pretrain_01/checkpoint-00069_features_event
    # output/pretrain_01/checkpoint-00079_features_event
    # output/pretrain_01/checkpoint-00089_features_event
    # output/pretrain_01/checkpoint-00099_features_event
    output/connectome_features_event
)

for prefix in ${PREFIXES[@]}; do
    outdir="${prefix}_probe"
    python main_task_linear_probe.py \
        --feat_prefix $prefix \
        --output_dir $outdir \
        --target $TARGET
done
