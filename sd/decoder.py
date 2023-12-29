import torch
from torch import nn

from attention import SelfAttention


class VAEResidualBlock(nn.Module):
    """Applies skip connections, group normalization and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # To ensure the output_channels and input_channels
        # are equal before we can add the skip connection
        self.residual_layer = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, in_channels, h, w)
        residual = x

        x = self.conv_1(
            self.silu(self.groupnorm_1(x))
        )  # (b, in_channels, h, w) -> (b, out_channels, h, w)
        x = self.conv_2(
            self.silu(self.groupnorm_2(x))
        )  # (b, out_channels, h, w) -> (b, out_channels, h, w)

        return self.residual_layer(residual) + x


class VAEAttentionBlock(nn.Module):
    """Applies the self-attention mechanism."""

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, channels, h, w)
        b, c, h, w = x.shape

        residual = x

        x = self.groupnorm(x)  # (b, channels, h, w) -> (b, channels, h, w)

        x = x.view(b, c, h * w)  # (b, channels, h, w) -> (b, channels, h * w)
        x = x.transpose(1, 2)  # (b, channels, h * w) -> (b, h * w, channels)
        x = self.attention(x)  # (b, h * w, channels) -> (b, h * w, channels)
        x = x.transpose(1, 2)  # (b, h * w, channels) -> (b, channels, h * w)
        x = x.view(b, c, h, w)  # (b, channels, h * w) -> (b, channels, h, w)

        return residual + x


class Decoder(nn.Sequential):
    """Decodes the latent vector (i.e output from UNet) into the generated image."""

    def __init__(self):
        super().__init__(
            # (b, 4, h / 8, w / 8) -> (b, 4, h / 8, w / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # (b, 4, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEAttentionBlock(512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # Repeats the rows and columns of the data by `scale_factor` (like when you resize an image by doubling its size).
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 4, w / 4)
            nn.Upsample(scale_factor=2),
            # (b, 512, h / 4, w / 4) -> (b, 512, h / 4, w / 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (b, 512, h / 4, w / 4) -> (b, 512, h / 4, w / 4)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 4, w / 4) -> (b, 512, h / 4, w / 4)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 4, w / 4) -> (b, 512, h / 4, w / 4)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 4, w / 4) -> (b, 512, h / 2, w / 2)
            nn.Upsample(scale_factor=2),
            # (b, 512, h / 2, w / 2) -> (b, 512, h / 2, w / 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (b, 512, h / 2, w / 2) -> (b, 256, h / 2, w / 2)
            VAEResidualBlock(512, 256),
            # (b, 256, h / 2, w / 2) -> (b, 256, h / 2, w / 2)
            VAEResidualBlock(256, 256),
            # (b, 256, h / 2, w / 2) -> (b, 256, h / 2, w / 2)
            VAEResidualBlock(256, 256),
            # (b, 256, h / 2, w / 2) -> (b, 256, h, w)
            nn.Upsample(scale_factor=2),
            # (b, 256, h, w) -> (b, 256, h, w)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (b, 256, h, w) -> (b, 128, h, w)
            VAEResidualBlock(256, 128),
            # (b, 128, h, w) -> (b, 128, h, w)
            VAEResidualBlock(128, 128),
            # (b, 128, h, w) -> (b, 128, h, w)
            VAEResidualBlock(128, 128),
            # (b, 128, h, w) -> (b, 128, h, w)
            nn.GroupNorm(32, 128),
            # (b, 128, h, w) -> (b, 128, h, w)
            nn.SiLU(),
            # (b, 128, h, w) -> (b, 3, h, w)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, 4, h / 8, w / 8) - latent vector

        # Nullify the scaling (see forward pass of Encoder)
        x /= 0.18215

        for module in self:
            x = module(x)

        # (b, 3, h, w)
        return x
