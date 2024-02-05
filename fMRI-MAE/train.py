# Training script for fMRI-MAE
import argparse
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
import yaml
from accelerate import Accelerator
from einops import rearrange
from models import SimpleViT
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DataPrepper
from utils import contrastive_loss as contrastive_loss_func
from utils import count_params, grayscale_decoder, numpy_decoder

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(config: dict):
    # Load the model config
    patch_size = config["patch_size"]
    frame_patch_size = config["frame_patch_size"]

    # Load the model
    model = SimpleViT(
        image_size=config["input_size"],  # depth, height, width
        image_patch_size=(
            patch_size,
            patch_size,
            patch_size,
        ),  # depth, height, width patch size
        frames=config["num_frames"],
        frame_patch_size=frame_patch_size,
        depth=config["depth"],
        heads=config["heads"],
        dim=config["dim"],
        mlp_dim=config[
            "mlp_dim"
        ],  # TODO: right now dim needs to equal mlp_dim, and both need to be 512
        channels=1,
        use_rope_emb=config["use_rope_emb"],
        use_cls_token=config["use_cls_token"],
    )
    count_params(model)
    return model


def get_optimizer_and_scheduler(
    model: nn.Module, max_lr: float, num_epochs: int, num_iterations_per_epoch: int
):
    # Load the optimizer and scheduler
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

    total_steps = num_epochs * num_iterations_per_epoch
    logger.info("total training steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1,
        pct_start=2 / num_epochs,
    )
    return optimizer, lr_scheduler


def get_data_loaders(config: dict):
    # load the train and test dataloaders
    train_urls = config["train_urls"]
    test_urls = config["test_urls"]

    def my_split_by_node(urls):
        return urls

    aug_transform = DataPrepper(
        masking_strategy="conservative",
        patch_depth=config["patch_size"],
        patch_height=config["patch_size"],
        patch_width=config["patch_size"],
        frame_patch_size=config["frame_patch_size"],
    )
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        num_devices = 1
    batch_size = int(config["global_batch_size"] / num_devices)

    if train_urls[:2] == "s3":
        train_urls = f"pipe:aws s3 cp {train_urls} -"
    logger.info(train_urls)
    train_data = (
        wds.WebDataset(
            train_urls,
            resampled=True,
            nodesplitter=my_split_by_node,
            cache_dir=config["cache_dir"],
            cache_size=config["cache_size"],
        )
        .shuffle(100)
        .rename(
            key="__key__",
            func="func.png",
            header="header.npy",
            dataset="dataset.txt",
            minmax="minmax.npy",
            meansd="meansd.png",
        )
        .map_dict(
            func=grayscale_decoder, meansd=grayscale_decoder, minmax=numpy_decoder
        )
        .to_tuple(*("func", "minmax", "meansd"))
        .map(aug_transform)
        .with_epoch(config["num_samples_per_epoch"])
    )
    train_dl = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    if test_urls[:2] == "s3":
        test_urls = f"pipe:aws s3 cp {test_urls} -"
    logger.info(test_urls)
    test_data = (
        wds.WebDataset(
            test_urls,
            resampled=False,
            nodesplitter=my_split_by_node,
            cache_dir=config["cache_dir"],
            cache_size=config["cache_size"],
        )
        .rename(
            key="__key__",
            func="func.png",
            header="header.npy",
            dataset="dataset.txt",
            minmax="minmax.npy",
            meansd="meansd.png",
        )
        .map_dict(
            func=grayscale_decoder,
            meansd=grayscale_decoder,
            minmax=numpy_decoder,
        )
        .to_tuple(*("func", "minmax", "meansd"))
        .map(aug_transform)
        .with_epoch(config["num_samples_per_epoch"])
    )
    test_dl = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    num_iterations_per_epoch: int = config["num_samples_per_epoch"] // batch_size
    num_patches = int(
        (config["input_size"][0] / config["patch_size"])
        * (config["input_size"][1] / config["patch_size"])
        * (config["input_size"][2] / config["patch_size"])
        * config["num_frames"]
        // config["frame_patch_size"]
    )
    return train_dl, test_dl, num_iterations_per_epoch, num_patches


def load_pretrained_model(
    model: nn.Module,
    optimizer: nn.Module,
    lr_scheduler: nn.Module,
    last_pt_path: Optional[str],
):
    epoch = 0
    recon_losses = []
    contrastive_losses = []
    test_losses = []
    lrs = []
    if last_pt_path is not None and os.path.exists(last_pt_path):
        state_dicts = torch.load(last_pt_path)

        model.load_state_dict(state_dicts["state_dict"])
        optimizer.load_state_dict(state_dicts["optimizer"])
        lr_scheduler.load_state_dict(state_dicts["scheduler"])
        # load the losses
        epoch = state_dicts["epoch"]
        recon_losses = state_dicts["recon_losses"]
        contrastive_losses = state_dicts["contrastive_losses"]
        test_losses = state_dicts["test_losses"]
        # load the learning rates
        lrs = state_dicts["lrs"]
    return (
        model,
        optimizer,
        lr_scheduler,
        epoch,
        recon_losses,
        contrastive_losses,
        test_losses,
        lrs,
    )


def get_encoder_decoder_masks(
    num_patches: int,
    num_frames: int,
    tube_mask_ratio: float,
    decoder_mask_ratio: float,
    brain_pos_pats: torch.Tensor,
):
    # create tube mask (i.e., a mask that is the same for all frames/timepoints)
    tube_mask = torch.zeros(num_patches // num_frames).to(torch.bool)
    batch_positive_approx = (
        brain_pos_pats[:, : num_patches // num_frames].float().mean(dim=0) > 0
    )
    mask_idx_candidates = torch.where(batch_positive_approx)[0]
    mask_idx_candidates = mask_idx_candidates[torch.randperm(len(mask_idx_candidates))]
    tube_idx = mask_idx_candidates[
        : int(num_patches / num_frames * (1 - tube_mask_ratio))
    ]
    tube_mask[tube_idx] = True
    tube_mask = tube_mask.tile(num_frames)

    # create decoder mask
    decoder_mask = torch.zeros(num_patches // num_frames).to(torch.bool)
    remaining_mask_idx = mask_idx_candidates[
        int(num_patches / num_frames * (1 - tube_mask_ratio)) :
    ]
    decoder_mask_idx = remaining_mask_idx[
        : int(num_patches / num_frames * (1 - decoder_mask_ratio))
    ]
    decoder_mask[decoder_mask_idx] = True
    decoder_mask = decoder_mask.tile(num_frames)

    return tube_mask, decoder_mask


def train(config_path: str):
    # Load the config and the parameters
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # class token config ###
    use_cls_token = config["use_cls_token"]

    # Loss Config
    use_contrastive_loss = config["use_contrastive_loss"]
    constrastive_loss_weight = config["constrastive_loss_weight"]
    use_cls_token = (
        True if use_contrastive_loss else use_cls_token
    )  # if using contrastive loss, we need to add a class token

    # Training Config
    save_per_epochs = config["save_per_epochs"]
    mixed_precision = config["mixed_precision"]
    accelerator = Accelerator(split_batches=False, mixed_precision=mixed_precision)

    # load the dataloaders, models, optimizers, and schedulers
    train_dl, test_dl, num_iterations_per_epoch, num_patches = get_data_loaders(config)
    model = load_model(config)
    optimizer, lr_scheduler = get_optimizer_and_scheduler(
        model=model,
        max_lr=config["max_lr"],
        num_epochs=config["num_epochs"],
        num_iterations_per_epoch=num_iterations_per_epoch,
    )

    # continue training from a previous checkpoint
    last_pt_path = os.path.join(config["save_dir"], "last.pt")
    (
        model,
        optimizer,
        lr_scheduler,
        epoch,
        recon_losses,
        contrastive_losses,
        test_losses,
        lrs,
    ) = load_pretrained_model(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        last_pt_path=last_pt_path,
    )
    model, optimizer, train_dl, test_dl, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dl,
        test_dl,
        lr_scheduler,
        device_placement=[True, True, True, True, True],
    )

    # Training Loop
    local_rank = os.getenv("RANK")
    local_rank = 0 if local_rank is None else int(local_rank)
    data_type = (
        torch.bfloat16
        if mixed_precision == "bf16"
        else torch.float16
        if mixed_precision == "fp16"
        else torch.float32
    )
    mse = nn.MSELoss()
    if use_contrastive_loss:
        logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )  # learned logit scale

    progress_bar = tqdm(range(epoch, config["num_epochs"]), disable=(local_rank != 0))

    for epoch in progress_bar:
        with torch.cuda.amp.autocast(dtype=data_type):
            model.train()
            for train_i, batch in enumerate(
                tqdm(
                    train_dl,
                    total=num_iterations_per_epoch,
                    disable=(local_rank != 0),
                    leave=False,
                )
            ):  # total samples in 1 epoch = train_dl.nsamples
                optimizer.zero_grad()

                func, meansd, brain_pos_pats = batch
                if (
                    use_contrastive_loss
                ):  # create positive pairs by duplicating the batch
                    func = torch.cat([func, func], dim=0)
                    meansd = torch.cat([meansd, meansd], dim=0)
                    brain_pos_pats = torch.cat([brain_pos_pats, brain_pos_pats], dim=0)
                func = func.unsqueeze(1)  # .to(device)
                tube_mask, decoder_mask = get_encoder_decoder_masks(
                    num_patches=num_patches,
                    num_frames=config["num_frames"],
                    tube_mask_ratio=config["tube_mask_ratio"],
                    decoder_mask_ratio=config["decoder_mask_ratio"],
                    brain_pos_pats=brain_pos_pats,
                )
                encoder_out = model(func, encoder_mask=tube_mask)
                if use_cls_token:
                    enc_cls_token = encoder_out[:, :1, :]
                decoder_out = model(
                    encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask
                )
                num_decoder_patches = int(
                    num_patches * (1 - config["decoder_mask_ratio"])
                )
                output = decoder_out[:, -num_decoder_patches:]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(func)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                target = target_patches_vit[:, decoder_mask]
                loss = mse(output, target)

                # contrastive loss
                if use_contrastive_loss:
                    n_b = len(func) // 2
                    cls_token1 = enc_cls_token[
                        :n_b, 0, :
                    ]  # first half of batch, cls_token shape B, 1, d_model
                    cls_token2 = enc_cls_token[n_b:, 0, :]
                    contrastive_loss = contrastive_loss_func(
                        cls_token1, cls_token2, temperature=logit_scale
                    )
                    loss += constrastive_loss_weight * contrastive_loss
                    contrastive_losses.append(contrastive_loss.item())

                accelerator.backward(loss)
                optimizer.step()
                recon_losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])
            logger.info(f"Train Epoch {epoch} completed")
            model.eval()
            for test_i, batch in enumerate(
                tqdm(
                    test_dl,
                    total=num_iterations_per_epoch,
                    disable=(local_rank != 0),
                    leave=False,
                )
            ):
                func, meansd, brain_pos_pats = batch
                func = func.unsqueeze(1)
                tube_mask, decoder_mask = get_encoder_decoder_masks(
                    num_patches=num_patches,
                    num_frames=config["num_frames"],
                    tube_mask_ratio=config["tube_mask_ratio"],
                    decoder_mask_ratio=config["decoder_mask_ratio"],
                    brain_pos_pats=brain_pos_pats,
                )
                # encode the tube patches
                encoder_out = model(func, encoder_mask=tube_mask)
                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask
                )
                # subset only the reconstructed decoder patches
                output = decoder_out[:, -num_decoder_patches:]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(func)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                target = target_patches_vit[:, decoder_mask]
                loss = mse(output, target)
                test_losses.append(loss.item())
            logs = {
                "train/loss": np.mean(recon_losses[-(train_i + 1) :]),
                "test/loss": np.mean(test_losses[-(test_i + 1) :]),
            }
            progress_bar.set_postfix(**logs)

            if epoch % save_per_epochs == 0:
                os.makedirs(config["save_dir"], exist_ok=True)
                # save the model state dict, optimizer state dict,
                # schedular state dict, and epoch number
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "recon_losses": recon_losses,
                        "contrastive_losses": contrastive_losses,
                        "test_losses": test_losses,
                        "lrs": lrs,
                    },
                    os.path.join(config["save_dir"], "last.pt"),
                )
                logger.info(
                    f"Model saved at {os.path.join(config['save_dir'], 'last.pt')}"
                )

    # save model ckpt and the configs
    if local_rank == 0:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "recon_losses": recon_losses,
                "contrastive_losses": contrastive_losses,
                "test_losses": test_losses,
                "lrs": lrs,
                "config": config,
            },
            os.path.join(config["save_dir"], "model.pt"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )

    args = parser.parse_args()
    train(args.config)
