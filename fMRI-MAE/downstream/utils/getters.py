from models import BrainNetwork
from meta import subject_to_voxels

def get_backbone(path):
    pass

def get_mlp_head(args):
    out_dim = 257 * 768
    num_voxels = subject_to_voxels[args.subj]
    voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim) 
    head = BrainNetwork(**voxel2clip_kwargs) # TODO: make the arguments of brian netwrok configurable using arg.
    return head