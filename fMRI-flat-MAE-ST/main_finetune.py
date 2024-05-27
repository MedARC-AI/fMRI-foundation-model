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

import models_vit

import util.lr_decay as lrd
import util.misc as misc

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import webdataset as wds
from iopath.common.file_io import g_pathmgr as pathmgr
from torch.utils.data import default_collate
from engine_finetune import evaluate, train_one_epoch
from util.hcp_flat import create_hcp_flat

from util.logging import master_print as print
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import set_requires_grad

from timm.loss import LabelSmoothingCrossEntropy
from timm.layers import trunc_normal_

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

PROJECT = "fMRI-foundation-model"
NUM_CLASSES = {"task": 14, "trial_type": 21}
CLIP_MODES = {"task": "seq", "trial_type": "event"}


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fMRI fine-tuning for task/state classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_patch16_fmri",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
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
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--freeze_params",
        default="",
        help="comma-separated list of patterns for params to freeze"
    )
    parser.add_argument(
        "--unfreeze_params",
        default="",
        help="comma-separated list of patterns for params to unfreeze"
    )
    parser.add_argument(
        "--global_pool",
        default="avg",
        choices=["avg", "cls", "spatial"],
        help="global pool mode"
    )
    parser.add_argument(
        "--target",
        default="trial_type",
        choices=["task", "trial_type"],
        help="classification target",
    )
    parser.add_argument(
        "--path_to_data_dir",
        type=str,
        default=None,
        help="local path for HCP-Flat data",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=100000,
        help="number of training samples per epoch",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=5000,
        help="number of training samples per epoch",
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
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
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
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # Video related configs
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.set_defaults(fp32=True)
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
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
        root=args.path_to_data_dir,
        split="train",
        clip_mode=CLIP_MODES[args.target],
        target=args.target,
        frames=args.num_frames,
        shuffle=True,
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
    num_train_batches = args.num_train_samples // (
        misc.get_world_size() * args.batch_size
    )
    data_loader_train = data_loader_train.with_epoch(num_train_batches)

    dataset_val = create_hcp_flat(
        root=args.path_to_data_dir,
        split="test",
        shards=range(87),  # first half of shards used as val
        clip_mode=CLIP_MODES[args.target],
        target=args.target,
        frames=args.num_frames,
        shuffle=False,
    )
    data_loader_val = wds.WebLoader(
        dataset_val.batched(
            args.batch_size, collation_fn=default_collate, partial=False
        ),
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    num_val_batches = args.num_val_samples // (
        misc.get_world_size() * args.batch_size
    )
    data_loader_val = data_loader_val.with_epoch(num_val_batches)

    if global_rank == 0 and args.output_dir:
        if args.name:
            args.output_dir = f"{args.output_dir}/{args.name}"
        Path(args.output_dir).mkdir(parents=True)

    model = models_vit.__dict__[args.model](
        num_classes=NUM_CLASSES[args.target],
        **vars(args),
    )

    if misc.get_last_checkpoint(args) is None and args.finetune and not args.eval:
        with pathmgr.open(args.finetune, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if "model" in checkpoint.keys():
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint["model_state"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if args.freeze_params:
        set_requires_grad(model, args.freeze_params.split(","), False)

    if args.unfreeze_params:
        set_requires_grad(model, args.unfreeze_params.split(","), True)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    print(f"trainable params:\n{trainable_params}")

    eff_batch_size = (
        args.batch_size * args.accum_iter * misc.get_world_size()
    )

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )
        model_without_ddp = model.module

    if global_rank == 0 and args.wandb:
        wandb.watch(model)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler(fp32=args.fp32)

    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(
            model,
            criterion,
            data_loader_val,
            device,
            epoch=args.start_epoch,
            args=args,
            fp32=args.fp32,
            num_batches=num_val_batches,
            log_wandb=args.wandb,
        )
        exit(0)

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            args=args,
            fp32=args.fp32,
            num_batches=num_train_batches,
            log_wandb=args.wandb,
        )
        if args.output_dir:
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(
            model,
            criterion,
            data_loader_val,
            device,
            epoch=epoch,
            args=args,
            fp32=args.fp32,
            num_batches=num_val_batches,
            log_wandb=args.wandb,
        )
        max_accuracy = max(max_accuracy, test_stats["acc"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return [checkpoint_path]


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
