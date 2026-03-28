import os
import sys
import torch
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.models import Generator
from src.utils import tensor_to_image


# ── Global setup ─────────────────────────────────────────────────────────────
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = None  # Smartphone -> DSLR
F = None  # DSLR -> Smartphone


def load_models(checkpoint_path: str):
    """Load both generators from a checkpoint file."""
    global G, F

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train the model first or download pretrained weights."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)

    G = Generator(config.IMG_CHANNELS,
                  config.N_RESIDUAL_BLOCKS).to(device)
    F = Generator(config.IMG_CHANNELS,
                  config.N_RESIDUAL_BLOCKS).to(device)

    G.load_state_dict(ckpt["G"])
    F.load_state_dict(ckpt["F"])

    G.eval()
    F.eval()

    print(f"[App] Models loaded from: {checkpoint_path}")
    print(f"[App] Device: {device}")


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    transform = T.Compose([
        T.Resize(config.IMG_SIZE),
        T.CenterCrop(config.IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std =[0.5, 0.5, 0.5]),
    ])
    return transform(pil_image).unsqueeze(0).to(device)


def enhance_to_dslr(input_image: np.ndarray) -> np.ndarray:
    """
    Gradio inference function: Smartphone photo -> DSLR quality.

    Args:
        input_image: numpy array from Gradio (H, W, 3) uint8

    Returns:
        Enhanced image as numpy array (H, W, 3) uint8
    """
    if G is None:
        raise gr.Error(
            "Model not loaded! Please provide a checkpoint path."
        )

    pil_img = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")
    tensor  = preprocess_image(pil_img)

    with torch.no_grad():
        enhanced = G(tensor)

    return tensor_to_image(enhanced[0])


def enhance_to_smartphone(input_image: np.ndarray) -> np.ndarray:
    """
    Gradio inference function: DSLR photo -> Smartphone style.

    Args:
        input_image: numpy array from Gradio (H, W, 3) uint8

    Returns:
        Smartphone-style image as numpy array
    """
    if F is None:
        raise gr.Error(
            "Model not loaded! Please provide a checkpoint path."
        )

    pil_img = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")
    tensor  = preprocess_image(pil_img)

    with torch.no_grad():
        converted = F(tensor)

    return tensor_to_image(converted[0])


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_interface():
    with gr.Blocks(
        title="Smartphone ↔ DSLR Enhancement — CycleGAN",
        theme=gr.themes.Soft(),
    ) as demo:

        # Header
        gr.Markdown("""
        # 📷 Smartphone ↔ DSLR Photo Enhancement
        ### Unpaired Image-to-Image Translation using CycleGAN (PyTorch)
        
        Upload a smartphone photo to enhance it to DSLR quality,
        or convert a DSLR photo to smartphone style.
        
        > **Paper:** Zhu et al., *Unpaired Image-to-Image Translation 
        using Cycle-Consistent Adversarial Networks*, ICCV 2017
        """)

        # Checkpoint loader
        with gr.Row():
            checkpoint_input = gr.Textbox(
                label       = "Checkpoint Path",
                placeholder = "./checkpoints/latest.pth",
                value       = "./checkpoints/latest.pth",
                scale       = 4,
            )
            load_btn = gr.Button("Load Model", variant="primary", scale=1)

        load_status = gr.Textbox(
            label     = "Model Status",
            value     = "Model not loaded",
            interactive = False,
        )

        def on_load_model(path):
            try:
                load_models(path)
                return f"Model loaded successfully from: {path} | Device: {device}"
            except Exception as e:
                return f"Error: {str(e)}"

        load_btn.click(
            fn      = on_load_model,
            inputs  = [checkpoint_input],
            outputs = [load_status],
        )

        gr.Markdown("---")

        # Tab 1: Smartphone -> DSLR
        with gr.Tab("📱 → 📷  Smartphone to DSLR"):
            gr.Markdown("### Upload a smartphone photo — get DSLR quality output")
            with gr.Row():
                input_phone = gr.Image(
                    label  = "Input: Smartphone Photo",
                    type   = "numpy",
                    height = 300,
                )
                output_dslr = gr.Image(
                    label  = "Output: DSLR Quality",
                    type   = "numpy",
                    height = 300,
                )
            enhance_btn = gr.Button(
                "Enhance to DSLR Quality ✨",
                variant = "primary",
                size    = "lg",
            )
            enhance_btn.click(
                fn      = enhance_to_dslr,
                inputs  = [input_phone],
                outputs = [output_dslr],
            )

        # Tab 2: DSLR -> Smartphone
        with gr.Tab("📷 → 📱  DSLR to Smartphone"):
            gr.Markdown("### Upload a DSLR photo — convert to smartphone style")
            with gr.Row():
                input_dslr   = gr.Image(
                    label  = "Input: DSLR Photo",
                    type   = "numpy",
                    height = 300,
                )
                output_phone = gr.Image(
                    label  = "Output: Smartphone Style",
                    type   = "numpy",
                    height = 300,
                )
            convert_btn = gr.Button(
                "Convert to Smartphone Style 📱",
                variant = "primary",
                size    = "lg",
            )
            convert_btn.click(
                fn      = enhance_to_smartphone,
                inputs  = [input_dslr],
                outputs = [output_phone],
            )

        # Footer
        gr.Markdown("""
        ---
        **Model Architecture:**
        - Generator: ResNet-9 (11.3M parameters)
        - Discriminator: PatchGAN 70×70
        - Loss: GAN Loss + Cycle Consistency Loss (λ=10) + Identity Loss

        **Dataset:** DPED — smartphone vs DSLR flower photos (unpaired)
        
        **GitHub:** [smartphone2dslr-cyclegan](https://github.com/sidrahrahim5-cmyk/smartphone2dslr-cyclegan)
        """)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        share       = True,   # Public link banata hai
        server_port = 7860,
        show_error  = True,
    )