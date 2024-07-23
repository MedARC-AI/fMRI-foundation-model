import io
from typing import Any, Optional, Union

import av
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from PIL import Image

plt.rcParams["figure.dpi"] = 150

_FC_COLORS = np.array(
    [
        [ 64,  80, 160],
        [ 64,  96, 176],
        [ 96, 192, 240],
        [144, 208, 224],
        [255, 255, 255],
        [240, 240,  96],
        [240, 208,  64],
        [224, 112,  64],
        [224,  64,  48],
    ],
    dtype=np.uint8
)

FC_CMAP = LinearSegmentedColormap.from_list("fc", _FC_COLORS / 255.0)
FC_CMAP.set_bad("gray")


def plot_mask_pred(
    target: torch.Tensor,
    im_masked: torch.Tensor,
    im_paste: torch.Tensor,
    img_mask: Optional[torch.Tensor] = None,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    nrow: int = 8,
):
    # N T H W C -> (N T) H W C
    target = target.flatten(0, 1)[:nrow].cpu().numpy()
    im_masked = im_masked.flatten(0, 1)[:nrow].cpu().numpy()
    im_paste = im_paste.flatten(0, 1)[:nrow].cpu().numpy()

    if img_mask is not None:
        img_mask = img_mask.cpu().numpy()
    else:
        img_mask = None

    H, W = target.shape[1:3]
    ploth = 2.0
    plotw = (W / H) * ploth
    nrow = len(target)
    ncol = 3
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(plotw * ncol, ploth * nrow), squeeze=False
    )

    for ii in range(nrow):
        plt.sca(axs[ii, 0])
        imshow(im_masked[ii], mean=mean, std=std, mask=img_mask)

        plt.sca(axs[ii, 1])
        imshow(im_paste[ii], mean=mean, std=std, mask=img_mask)

        plt.sca(axs[ii, 2])
        imshow(target[ii], mean=mean, std=std, mask=img_mask)

    plt.tight_layout(pad=0.25)
    return fig


def video_mask_pred(
    target: torch.Tensor,
    im_masked: torch.Tensor,
    im_paste: torch.Tensor,
    img_mask: Optional[torch.Tensor] = None,
    file: Optional[io.FileIO] = None,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    cmap: Union[str, Colormap, None] = FC_CMAP,
    fps: float = 4.0,
) -> np.ndarray:
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # N T H W C -> (N T) H W C
    target = target.flatten(0, 1).cpu().numpy()
    im_masked = im_masked.flatten(0, 1).cpu().numpy()
    im_paste = im_paste.flatten(0, 1).cpu().numpy()

    # apply mask
    if img_mask is not None:
        img_mask = img_mask.unsqueeze(-1).cpu().numpy() > 0
        target = np.where(img_mask, target, float("nan"))
        im_masked = np.where(img_mask, im_masked, float("nan"))
        im_paste = np.where(img_mask, im_paste, float("nan"))

    # stack images
    stacked = np.concatenate([im_masked, im_paste, target], axis=-2)

    # scale values to [0, 1]
    if mean is not None:
        stacked = unscale(stacked, mean, std)
    else:
        vmin = target.min()
        vmax = target.max()
        stacked = (stacked - vmin) / (vmax - vmin)

    # apply colormap
    if cmap is not None:
        stacked = cmap(stacked.squeeze(-1))
        stacked = stacked[..., :3]  # drop alpha channel

    # to RGB
    stacked = (255 * stacked).astype(np.uint8)
    if file is not None:
        write_video(stacked, file, fps=fps)
    return stacked


def video_denoise(
    target: torch.Tensor,
    pred: torch.Tensor,
    img_mask: Optional[torch.Tensor],
    file: Optional[io.FileIO] = None,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    cmap: Union[str, Colormap, None] = FC_CMAP,
    fps: float = 4.0,
) -> np.ndarray:
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # N T H W C -> (N T) H W C
    target = target.flatten(0, 1).cpu().numpy()
    im_masked = im_masked.flatten(0, 1).cpu().numpy()
    im_paste = im_paste.flatten(0, 1).cpu().numpy()

    # apply mask
    if img_mask is not None:
        img_mask = img_mask.unsqueeze(-1).cpu().numpy() > 0
        target = np.where(img_mask, target, float("nan"))
        pred = np.where(img_mask, pred, float("nan"))

    # stack images
    stacked = np.concatenate([target, pred], axis=-2)

    # scale values to [0, 1]
    if mean is not None:
        stacked = unscale(stacked, mean, std)
    else:
        vmin = target.min()
        vmax = target.max()
        stacked = (stacked - vmin) / (vmax - vmin)

    # apply colormap
    if cmap is not None:
        stacked = cmap(stacked.squeeze(-1))
        stacked = stacked[..., :3]  # drop alpha channel

    # to RGB
    stacked = (255 * stacked).astype(np.uint8)
    if file is not None:
        write_video(stacked, file, fps=fps)
    return stacked


def imshow(
    image: np.ndarray,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs,
):
    # image: (H, W, C)
    assert image.shape[2] in (1, 3)
    if image.shape[2] == 1:
        kwargs = {
            "cmap": "gray",
            "vmin": 0.0,
            "vmax": 1.0,
            "interpolation": "nearest",
            **kwargs,
        }
    if mean is not None:
        image = unscale(image, mean, std)
    else:
        vmin, vmax = np.nanmin(image), np.nanmax(image)
        image = (image - vmin) / (vmax - vmin)
    if mask is not None:
        image = mask[..., None] * image
    plt.imshow(image, **kwargs)
    plt.axis("off")


def fig2pil(fig: Figure, format: str = "png") -> Image.Image:
    with io.BytesIO() as f:
        fig.savefig(f, format=format)
        f.seek(0)
        img = Image.open(f)
        img.load()
    return img


def write_video(
    frames: np.ndarray, file: io.FileIO, fps: float = 30.0, crf: int = 17
) -> None:
    assert frames.ndim == 4, "expected 4d input N H W C"
    assert frames.shape[-1] == 3, "expected RGB input"
    H, W = frames.shape[1:3]

    container = av.open(file, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf)}

    for frame_data in frames:
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        packet = stream.encode(frame)
        container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    container.close()


def unscale(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean)
    std = np.asarray(std)
    x = np.clip(x * std + mean, 0.0, 1.0)
    return x
