# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import util.misc as misc

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models_mae
import webdataset as wds
from iopath.common.file_io import g_pathmgr as pathmgr
from torch.utils.data import default_collate
from engine_pretrain import train_one_epoch
from util.hcp_flat import create_hcp_flat
from util.logging import master_print as print
from util.misc import NativeScalerWithGradNormCount as NativeScaler

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

PROJECT = "fMRI-foundation-model"


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fMRI pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_base_patch16_fmri",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--path_to_data_dir",
        type=str,
        default=None,
        help="local path for HCP-Flat data",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--name", type=str, default=None, help="name for current run",
    )
    parser.add_argument("--wandb", action="store_true", help="enable wandb logging")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")

    # Video related configs
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200000,
        help="number of samples per training epoch",
    )
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
    )
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--trunc_init",
        action="store_true",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.add_argument(
        "--no_fp32",
        action="store_false",
        dest="fp32",
    )
    parser.set_defaults(fp32=True)
    parser.add_argument(
        "--beta",
        default=None,
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--pred_t_dim",
        type=int,
        default=8,
    )
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(args):
    misc.init_distributed_mode(args)

    global_rank = misc.get_rank()
    if global_rank == 0 and args.wandb:
        assert has_wandb, "wandb not installed"
        wandb.init(project=PROJECT, name=args.name, config=vars(args))

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = create_hcp_flat(
        root=args.path_to_data_dir, split="train", frames=args.num_frames
    )
    data_loader_train = wds.WebLoader(
        dataset_train.batched(
            args.batch_size, collation_fn=default_collate, partial=False
        ),
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    num_batches = args.num_samples // (misc.get_world_size() * args.batch_size)
    data_loader_train = data_loader_train.with_epoch(num_batches)

    if global_rank == 0 and args.output_dir:
        if args.name:
            args.output_dir = f"{args.output_dir}/{args.name}"
        Path(args.output_dir).mkdir(parents=True)

    # define the model
    model = models_mae.__dict__[args.model](
        **vars(args),
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module

    if global_rank == 0 and args.wandb:
        wandb.watch(model)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )
    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = args.beta
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args=args,
            fp32=args.fp32,
            num_batches=num_batches,
            log_wandb=args.wandb,
        )
        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.debug:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
