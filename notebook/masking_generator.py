import numpy as np
import torch

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio, max_num_patches):
        self.mask_ratio = mask_ratio
        self.max_num_patches = max_num_patches
        self.frames, self.x, self.y, self.z = input_size
        self.num_patches_per_frame =  max_num_patches // self.frames
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self, token_shape):
        num_patches_per_frame = token_shape[1] * token_shape[2] * token_shape[3]
        frames = token_shape[0]
        num_masks_per_frame = int(self.mask_ratio * num_patches_per_frame)
        mask_per_frame = np.hstack([
            np.zeros(num_patches_per_frame - num_masks_per_frame),
            np.ones(num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask_per_frame = np.hstack([
            mask_per_frame,
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame - (num_patches_per_frame - num_masks_per_frame)),
            np.ones(self.num_masks_per_frame - num_masks_per_frame),
        ])
        mask = np.tile(mask_per_frame, (frames,1)).flatten()
        return torch.tensor(mask)
    
class AgnosticMaskingGenerator:
    def __init__(self, input_size, mask_ratio, max_num_patches):
        self.mask_ratio = mask_ratio
        self.max_num_patches = max_num_patches
        self.frames, self.x, self.y, self.z = input_size
        self.num_patches_per_frame =  max_num_patches // self.frames
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str
    
    def __call__(self, token_shape):
        num_patches_per_frame = token_shape[1] * token_shape[2] * token_shape[3]
        frames = token_shape[0]
        num_masks_per_frame = int(self.mask_ratio * num_patches_per_frame)
        mask = []
        for i in range(frames):
            mask_per_frame = np.hstack([
                np.zeros(num_patches_per_frame - num_masks_per_frame),
                np.ones(num_masks_per_frame),
            ])
            np.random.shuffle(mask_per_frame)
            mask_per_frame = np.hstack([
                mask_per_frame,
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame - (num_patches_per_frame - num_masks_per_frame)),
                np.ones(self.num_masks_per_frame - num_masks_per_frame),
            ])
            mask.append(mask_per_frame)
        mask = np.vstack(mask).flatten()
        return torch.tensor(mask)