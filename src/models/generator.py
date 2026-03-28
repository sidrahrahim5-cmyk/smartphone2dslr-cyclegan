import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A single residual block used inside the generator.
    Paper: Johnson et al. architecture with instance normalization.
    
    Structure: Conv -> InstanceNorm -> ReLU -> Conv -> InstanceNorm
    Input is added to output (skip connection).
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN Generator based on Johnson et al. (2016).

    Architecture for 256x256 images (9 residual blocks):
        c7s1-64 -> d128 -> d256 -> R256 x9 -> u128 -> u64 -> c7s1-3

    Where:
        c7s1-k : 7x7 Conv-InstanceNorm-ReLU, k filters, stride 1
        dk      : 3x3 Conv-InstanceNorm-ReLU, k filters, stride 2 (downsample)
        Rk      : Residual block, k filters
        uk      : 3x3 Fractional-strided Conv, k filters, stride 1/2 (upsample)
    """

    def __init__(self, input_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()

        # --- Initial convolution block ---
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # --- Downsampling (encoder) ---
        in_channels = 64
        for _ in range(2):
            out_channels = in_channels * 2
            layers += [
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # --- Residual blocks (transformation) ---
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(in_channels))

        # --- Upsampling (decoder) ---
        for _ in range(2):
            out_channels = in_channels // 2
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # --- Output convolution ---
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, input_channels, kernel_size=7, padding=0),
            nn.Tanh(),  # Output range: [-1, 1]
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Quick test - verify output shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gen = Generator(input_channels=3, num_residual_blocks=9).to(device)
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = gen(dummy_input)
    
    total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    
    print("=" * 45)
    print("Generator Architecture Test")
    print("=" * 45)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Parameters   : {total_params:,}")
    print(f"Device       : {device}")
    
    assert output.shape == dummy_input.shape, "Shape mismatch!"
    print("Shape check  : PASSED")
    print("=" * 45)