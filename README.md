# 📷 Smartphone → DSLR Photo Enhancement using CycleGAN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv%201703.10593-red)](https://arxiv.org/abs/1703.10593)

> Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
> **Zhu et al., ICCV 2017** — implemented in PyTorch with a custom Smartphone↔DSLR dataset

---

## 🎯 What This Project Does

This project implements **CycleGAN** to enhance smartphone photos to professional
DSLR quality — **without any paired training data**.

| Input (Smartphone) | Output (DSLR Quality) |
|---|---|
| Low dynamic range | High dynamic range |
| Noise, blur | Sharp, clean details |
| Small aperture (deep DoF) | Shallow depth of field |

---

## 🧠 Model Architecture
```
Generator (ResNet-9):
  Input (3×256×256)
  → Encoder: Conv7 → DownConv → DownConv
  → Transform: 9× Residual Blocks
  → Decoder: UpConv → UpConv → Conv7
  → Output (3×256×256)

Discriminator (PatchGAN 70×70):
  C64 → C128 → C256 → C512 → Output patch map
```

**Loss Function:**
```
L_total = L_GAN(G) + L_GAN(F) + λ·L_cycle + λ·0.5·L_identity
                                  λ = 10
```

---

## 📁 Project Structure
```
smartphone2dslr-cyclegan/
├── src/
│   ├── models/
│   │   ├── generator.py       # ResNet-9 Generator
│   │   ├── discriminator.py   # PatchGAN Discriminator
│   │   └── losses.py          # GAN + Cycle + Identity Loss
│   ├── data/
│   │   └── dataset.py         # Unpaired dataset loader
│   └── utils/
│       ├── replay_buffer.py   # 50-image history buffer
│       └── image_utils.py     # Visualization helpers
├── notebooks/
│   └── CycleGAN_Training.ipynb  # Google Colab training
├── app/
│   └── gradio_app.py          # Interactive web demo
├── config.py                  # All hyperparameters
├── train.py                   # Training loop
├── test.py                    # Single image inference
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sidrahrahim5-cmyk/smartphone2dslr-cyclegan.git
cd smartphone2dslr-cyclegan
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Test on a single image
```bash
python test.py --image your_photo.jpg --checkpoint checkpoints/latest.pth
```

### 5. Launch web demo
```bash
python app/gradio_app.py
```

---

## 🏋️ Training

### Option A: Google Colab (Recommended — Free GPU)
Open `notebooks/CycleGAN_Training.ipynb` in Google Colab.

### Option B: Local (GPU required)
```bash
python train.py
```

**Hyperparameters** (from paper):

| Parameter | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 1 |
| Learning rate | 0.0002 |
| Epochs | 200 |
| λ (cycle loss) | 10 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Replay buffer | 50 images |

---

## Dataset

**Oxford Flowers102** — processed into two unpaired domains

- **Domain A (Smartphone style):** Flowers102 with blur, noise, reduced contrast
- **Domain B (DSLR style):** Flowers102 with sharpening, enhanced contrast, rich colors
- **Training:** ~900 images per domain
- **Source:** `torchvision.datasets.Flowers102` (no manual download needed)
- **Unpaired:** CycleGAN does not require paired examples
---

## 📄 Citation
```bibtex
@inproceedings{CycleGAN2017,
  title     = {Unpaired Image-to-Image Translation using
               Cycle-Consistent Adversarial Networks},
  author    = {Zhu, Jun-Yan and Park, Taesung and
               Isola, Phillip and Efros, Alexei A.},
  booktitle = {ICCV},
  year      = {2017}
}
```

---

## 👩‍💻 Author

**Sidra Rahim** — AI Engineer  
GitHub: [@sidrahrahim5-cmyk](https://github.com/sidrahrahim5-cmyk)

---

*Implemented as part of Deep Learning coursework —
studying CycleGAN paper and applying it to a custom dataset.*