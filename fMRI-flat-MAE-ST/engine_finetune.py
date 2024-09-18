# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import util.lr_sched as lr_sched
import util.misc as misc
import torch
import wandb
from util.logging import master_print as print
from timm.utils import accuracy


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    args=None,
    fp32=False,
    num_batches=None,
    log_wandb=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    log_wandb = misc.is_main_process() and log_wandb

    accum_iter = args.accum_iter
    if num_batches is None:
        num_batches = len(data_loader)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(
            data_loader, print_freq, header, total_steps=num_batches
        )
    ):
        samples, targets = batch["image"], batch["target"]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / num_batches + epoch, args
            )

        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.view(b * r, c, t, h, w)
            targets = targets.view(b * r)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_wandb and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / num_batches + epoch) * 1000)
            wandb.log({"train_loss": loss_value_reduce, "lr": max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    args=None,
    fp32=False,
    num_batches=None,
    log_wandb=False,
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test: [{}]".format(epoch)
    log_wandb = misc.is_main_process() and log_wandb

    # switch to evaluation mode
    model.eval()

    if num_batches is None:
        num_batches = len(data_loader)

    for batch in metric_logger.log_every(
        data_loader, 10, header, total_steps=num_batches
    ):
        images, target = batch["image"], batch["target"]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if len(images.shape) == 6:
            b, r, c, t, h, w = images.shape
            images = images.view(b * r, c, t, h, w)
            target = target.view(b * r)

        # compute output
        with torch.cuda.amp.autocast(enabled=not fp32):
            output = model(images)
            loss = criterion(output, target)

        acc = accuracy(output, target)[0]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc"].update(acc.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if log_wandb:
        epoch_1000x = (epoch + 1) * 1000
        wandb.log(
            {
                "test_acc": metric_logger.acc.global_avg,
                "test_loss": metric_logger.loss.global_avg,
            },
            step=epoch_1000x,
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
