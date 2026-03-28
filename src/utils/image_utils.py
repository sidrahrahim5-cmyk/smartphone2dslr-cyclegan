import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def tensor_to_image(tensor):
    """
    [-1,1] tensor → numpy image [0,255] uint8
    """
    img = tensor.detach().cpu().clone()
    img = img * 0.5 + 0.5        # [-1,1] → [0,1]
    img = img.clamp(0, 1)
    img = img.numpy().transpose(1, 2, 0)  # CHW → HWC
    return (img * 255).astype(np.uint8)


def save_sample_grid(real_A, fake_B, rec_A,
                     real_B, fake_A, rec_B,
                     save_path, epoch, batch):
    """
    6 images ka grid save karo:
    real_A | fake_B | rec_A
    real_B | fake_A | rec_B
    """
    imgs = [real_A, fake_B, rec_A, real_B, fake_A, rec_B]
    imgs = [t[0] for t in imgs]  # batch dim hatao

    grid = make_grid(
        [torch.tensor(i).permute(2, 0, 1)
         for i in [tensor_to_image(t) for t in imgs]],
        nrow=3,
        normalize=False
    )

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title(
        f"Epoch {epoch} | Batch {batch}\n"
        "Row 1: Phone | Fake DSLR | Reconstructed Phone\n"
        "Row 2: DSLR  | Fake Phone | Reconstructed DSLR",
        fontsize=12
    )

    os.makedirs(save_path, exist_ok=True)
    fname = os.path.join(save_path, f"epoch{epoch:03d}_batch{batch:05d}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=100)
    plt.close()
    return fname


def denormalize(tensor):
    """[-1,1] → [0,1] for display."""
    return tensor * 0.5 + 0.5