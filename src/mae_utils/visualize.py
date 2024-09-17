import io
from typing import Any, Optional

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

plt.rcParams["figure.dpi"] = 150


def plot_mask_pred(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    nrow: int = 8,
):
    # imgs: [N, C, T, H, W]
    # pred: [N, t*h*w, u*p*p*C]
    # mask: [N, t*h*w], 0 is keep, 1 is remove,
    target = torch.index_select(
        imgs,
        2,
        torch.linspace(
            0,
            imgs.shape[2] - 1,
            model.pred_t_dim,
        )
        .long()
        .to(imgs.device),
    )
    target = torch.einsum("ncthw->nthwc", target)
    target = target.flatten(0, 1)[:nrow].cpu()
    
    mask = mask.unsqueeze(-1).repeat(
        1, 1, pred.shape[-1]
    )  # (N, T*H*W, p*p*c)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("ncthw->nthwc", mask).cpu()
    mask = mask.flatten(0, 1)[:nrow].cpu()

    pred = pred.detach()
    pred = model.unpatchify(pred)
    pred = torch.einsum("ncthw->nthwc", pred).cpu()
    pred = pred.flatten(0, 1)[:nrow].cpu()

    # masked image
    im_masked = target * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = target * (1 - mask) + pred * mask

    if model.img_mask is not None:
        img_mask = model.img_mask.cpu()
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


def imshow(
    image: torch.Tensor,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    mask: Optional[torch.Tensor] = None,
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
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        image = torch.clip(image * std + mean, 0.0, 1.0)
    if mask is not None:
        image = mask.unsqueeze(-1) * image
    plt.imshow(image, **kwargs)
    plt.axis("off")


def fig2pil(fig: Figure, format: str = "png") -> Image.Image:
    with io.BytesIO() as f:
        fig.savefig(f, format=format)
        f.seek(0)
        img = Image.open(f)
        img.load()
    return img
