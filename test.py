import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

from config import Config
from src.models import Generator
from src.utils  import tensor_to_image


def load_generator(checkpoint_path, device, config):
    """Load trained generator from checkpoint."""
    G = Generator(config.IMG_CHANNELS,
                  config.N_RESIDUAL_BLOCKS).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    print(f"[Info] Loaded checkpoint: {checkpoint_path}")
    return G


def preprocess(image_path, img_size=256):
    """Load and preprocess a single image."""
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std =[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # Add batch dimension


def enhance_photo(image_path, checkpoint_path, output_path=None):
    """
    Translate a smartphone photo to DSLR quality.

    Args:
        image_path      : path to input smartphone photo
        checkpoint_path : path to trained model checkpoint
        output_path     : where to save output (optional)
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # Load model
    G = load_generator(checkpoint_path, device, config)

    # Preprocess input
    input_tensor = preprocess(image_path, config.IMG_SIZE).to(device)

    # Generate enhanced image
    with torch.no_grad():
        output_tensor = G(input_tensor)

    # Convert to images
    input_img  = tensor_to_image(input_tensor[0])
    output_img = tensor_to_image(output_tensor[0])

    # Display side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_img)
    axes[0].set_title("Input: Smartphone Photo", fontsize=13,
                       fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(output_img)
    axes[1].set_title("Output: DSLR Quality (CycleGAN)",
                       fontsize=13, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("Smartphone → DSLR Enhancement", fontsize=15)
    plt.tight_layout()

    # Save output
    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base}_enhanced.png"

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"[Saved] Result: {output_path}")
    plt.show()

    return output_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test CycleGAN: Smartphone -> DSLR Enhancement"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input smartphone image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/latest.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output image"
    )
    args = parser.parse_args()

    enhance_photo(args.image, args.checkpoint, args.output)