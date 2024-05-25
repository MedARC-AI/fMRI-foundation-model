import argparse
import datetime
import os
import random
import time
from pathlib import Path
from typing import Iterable

import util.misc as misc

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models_vit
import webdataset as wds
from elbow.sinks import BufferedParquetWriter
from util.connectivity import Connectome
from util.hcp_flat import create_hcp_flat
from util.misc import setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fMRI feature extraction", add_help=False)
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path where to save",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_patch16_fmri",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--no_qkv_bias", action="store_true")
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
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    parser.add_argument("--ckpt_path", default="", help="load from checkpoint")

    parser.add_argument(
        "--path_to_data_dir",
        type=str,
        default=None,
        help="local path for HCP-Flat data",
    )
    parser.add_argument("--split", default="train", help="dataset split")
    parser.add_argument(
        "--clip_mode",
        default="seq",
        choices=["seq", "event"],
        help="clip mode",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="number of samples to extract",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU",
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):
    setup_for_distributed(True)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset = create_hcp_flat(
        root=args.path_to_data_dir,
        split=args.split,
        clip_mode=args.clip_mode,
        frames=args.num_frames,
        shuffle=False,
    )
    data_loader = wds.WebLoader(
        dataset.batched(args.batch_size, partial=False),
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    if args.num_samples:
        num_batches = args.num_samples // args.batch_size
        data_loader = data_loader.with_epoch(num_batches)
    else:
        # dummy number of batches if extracting the full dataset
        num_batches = 99999999

    output_path = Path(args.output_path)
    if output_path.exists():
        raise FileExistsError(f"output_path {output_path} already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # define the model
    if args.model == "connectome":
        model = Connectome()
    else:
        model = models_vit.__dict__[args.model](
            **vars(args),
        )

    if args.ckpt_path:
        print("Load pre-trained checkpoint from: %s" % args.ckpt_path)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")

        # load pre-trained model
        msg = model.load_state_dict(ckpt["model"], strict=False)
        print(msg)

    model.to(device)

    print("Start extract")
    start_time = time.time()

    with BufferedParquetWriter(output_path, blocking=True) as writer:
        for sample in extract_features(
            model,
            data_loader,
            device,
            args,
            fp32=args.fp32,
            num_batches=num_batches,
        ):
            writer.write(sample)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Extract time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    args=None,
    fp32=False,
    num_batches=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    if num_batches is None:
        num_batches = len(data_loader)

    for samples, samples_meta in metric_logger.log_every(
        data_loader, 20, total_steps=num_batches
    ):

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=not fp32):
            features = model.forward_features(samples)

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())

        features = features.cpu().numpy()
        for feat, meta in zip(features, samples_meta):
            yield {"feature": feat, **meta}


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
