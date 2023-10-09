import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse

from .utils import get_backbone, get_mlp_head
from .utils.models import Clipper

parser = argparse.ArgumentParser(description="Retrival")
parser.add_argument(
    "--backbone-path", 
    type=str,
    help="Path for the backbone encoder (foundation model).",
)
parser.add_argument(
    "--head-path", 
    type=str,
    help="Path for head that takes backbone latents.",
)
parser.add_argument(
    "--data-path",
    type=str,
    default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
    help="Path to where NSD data is stored (see README)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    help="Directory path for outputs",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=300,
    help="Number of batches in iteration",
)
parser.add_argument(
    "--checkpoint-frequency",
    type=int,
    default=10,
    help="Number of epochs after each checkpoint",
)
parser.add_argument(
    "--finetune-backbone",
    type=bool,
    default=False,
    help="Whether or not to finetune the backbone",
)
parser.add_argument(
    "--subj",
    type=int,
    default=1,
    choices=[1,2,5,7],
)

def train_retrival(
    backbone,
    head,
    dataset,
    epochs,
    optimizer,
    scheduler,
    output_dir,
):
    for i, (voxel, img, coco) in enumerate(tqdm(dataset, total=epochs)):
        pass
    pass

def run_retrival_eval(
        backbone,
        head,
        output_dir,
        data_path,
        batch_size,
        epochs,
        checkpoint_frequency,
        finetune,
        ):
    seed = 0
    clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True).cuda()

    # Dataset
    voxels_key = 'nsdgeneral.npy'
    dataset = wds.WebDataset(data_path, resampled=True)\
        .decode("torch") \
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy") \
        .to_tuple("voxels", "images", "coco") \
        .batched(batch_size, partial=False) \
        .with_epoch(epochs)

    # Training utils TODO: make configurable
    optimizer_parms = list(backbone.parameters()) + list(head.parameters()) if finetune else head.parameters()
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(optimizer_parms, lr=learning_rate, momentum=0.9, weight_decay=0) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    # TODO: add checkpoint

    result = train_retrival(
        backbone=backbone,
        head=head,
        dataset=dataset,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
    )
    pass


def main(args):
    backbone = get_backbone(args.backbone_path)
    head = get_mlp_head(args)
    run_retrival_eval(
        backbone=backbone,
        head=head,
        output_dir=args.output_dir,
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_frequency=args.checkpoint_frequenc,
        finetune=args.finetune_backbone
    )
    return None