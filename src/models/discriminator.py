import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    CycleGAN Discriminator — 70x70 PatchGAN.

    Instead of classifying the whole image as real/fake,
    PatchGAN classifies overlapping 70x70 patches.
    
    Benefits:
        - Fewer parameters than full-image discriminator
        - Works on any image size (fully convolutional)
        - Captures high-frequency texture details better

    Architecture:
        C64 -> C128 -> C256 -> C512 -> output
        
    Where Ck = Conv(4x4, stride=2) -> InstanceNorm -> LeakyReLU(0.2)
    Note: No InstanceNorm on first layer (C64).
    """

    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_ch, out_ch, normalize=True):
            layers = [
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # C64 — no normalization on first layer
            *discriminator_block(input_channels, 64,  normalize=False),
            # C128
            *discriminator_block(64,  128),
            # C256
            *discriminator_block(128, 256),
            # C512 — stride=1 on last conv block
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output — 1 channel patch map (real=1, fake=0)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disc = Discriminator(input_channels=3).to(device)

    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = disc(dummy_input)

    total_params = sum(p.numel() for p in disc.parameters()
                       if p.requires_grad)

    print("=" * 45)
    print("Discriminator Architecture Test")
    print("=" * 45)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Parameters   : {total_params:,}")
    print(f"Device       : {device}")
    print("=" * 45)