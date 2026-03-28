import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def tensor_to_image(tensor):
    """
    Convert a normalized tensor [-1, 1] to a numpy image [0, 255].
    """
    img = tensor.detach().cpu().clone()
    img = img * 0.5 + 0.5        # [-1, 1] -> [0, 1]
    img = img.clamp(0, 1)
    img = img.numpy().transpose(1, 2, 0)   # CHW -> HWC
    return (img * 255).astype(np.uint8)


def save_sample_grid(real_A, fake_B, rec_A,
                     real_B, fake_A, rec_B,
                     save_dir, epoch, batch_idx):
    """
    Save a 2x3 grid of images showing the translation cycle.

    Row 1: real_A (phone) | fake_B (fake DSLR) | rec_A (reconstructed phone)
    Row 2: real_B (DSLR)  | fake_A (fake phone) | rec_B (reconstructed DSLR)
    """
    images = [real_A, fake_B, rec_A,
              real_B, fake_A, rec_B]

    # Take first image from each batch
    images = [tensor_to_image(t[0]) for t in images]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    titles = [
        "Real Phone",    "Fake DSLR",         "Reconstructed Phone",
        "Real DSLR",     "Fake Phone",         "Reconstructed DSLR",
    ]

    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"Epoch {epoch:03d} | Batch {batch_idx:05d}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(
        save_dir, f"epoch{epoch:03d}_batch{batch_idx:05d}.png"
    )
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"[Saved] {path}")
    return path


def denormalize(tensor):
    """[-1, 1] -> [0, 1] for visualization."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)