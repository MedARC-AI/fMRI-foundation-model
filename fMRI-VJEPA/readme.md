# VJEPA Implementation

The goal of this subfolder is to train a foundational model for fMRI data built on the VJEPA architecture released by MetaAI:
https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/

## Current State (Changes from VidMAE V2)

The SimpleViT model currently used in VidMAE is revised to support three transformers:

1. X-Encoder
    Encodes context tokens (unmasked portion of data)
2. Y-Encoder
    Encodes entire dataset and outputs only the target tokens (masked portion of the data)
3. Predictor
    Encodes output of x-encoder + learnable masked tokens and outputs predicted target tokens
    
The model is trained on an L1 loss comparing the outputs of the Y-encoder and the Predictor. The Y-Encoder is trained through an exponential moving average of the X-Encoder (using a momentum scheduler) to prevent representation collapse (see paper Section 3.1).

## Results

The model has not yet been trained but this will be updated when it is.

## Contributing

Pull requests are welcome. The majority of the preprocessing metrics are identical to the MAE implementation and it would be useful to compare to the VJEPA codebase to find improvements (specifically in masking where VJEPA used a concept of short-range/long-range masks).