import torch
from decoder import VAEAttentionBlock, VAEResidualBlock
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Sequential):
    """Encodes the input noise or image into a latent vector"""

    def __init__(self):
        super().__init__(
            # (b, 3, h, w) -> (b, 128, h, w)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (b, 128, h, w) -> (b, 128, h, w)
            VAEResidualBlock(128, 128),
            # (b, 128, h, w) -> (b, 128, h, w)
            VAEResidualBlock(128, 128),
            # (b, 128, h, w) -> (b, 128, h / 2, w / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (b, 128, h / 2, w / 2) -> (b, 256, h / 2, w / 2)
            VAEResidualBlock(128, 256),
            # (b, 256, h / 2, w / 2) -> (b, 256, h / 2, w / 2)
            VAEResidualBlock(256, 256),
            # (b, 256, h / 2, w / 2) -> (b, 256, h / 4, w / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (b, 256, h / 4, w / 4) -> (b, 512, h / 4, w / 4)
            VAEResidualBlock(256, 512),
            # (b, 512, h / 4, w / 4) -> (b, 512, h / 4, w / 4)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 4, w / 4) -> (b, 512, h / 8, w / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEAttentionBlock(512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            VAEResidualBlock(512, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            nn.GroupNorm(32, 512),
            # (b, 512, h / 8, w / 8) -> (b, 512, h / 8, w / 8)
            nn.SiLU(),
            # (b, 512, h / 8, w / 8) -> (b, 8, h / 8, w / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (b, 8, h / 8, w / 8) -> (b, 8, h / 8, w / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, 3, h, w)
        # noise.shape: (b, 8, h / 8, w / 8) - same shape as output of encoder
        for module in self:
            # Perform manual padding on these convolutions where stride=2
            if getattr(module, "stride", None) == (2, 2):
                # (padding_left, padding_right, padding_top, padding_bottom)
                # (b, 3, h, w) -> (b, 3, h + padding_top + padding_bottom, w + padding_left + padding_right)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # Output of VAE encoder is the mean and log of the variance so we
        # can split across the channels (second dimension) into 2 chunks
        mean, log_var = torch.chunk(
            x, 2, dim=1
        )  # (b, 8, h / 8, w / 8) -> 2 x (b, 4, h / 8, w / 8)

        # Clamps the log variance, so that the variance is between (circa) 1e-14 and 1e8.
        log_var = torch.clamp(log_var, -30, 20)  # (b, 4, h / 8, w / 8)
        # Transform log variance into variance
        var = log_var.exp()  # (b, 4, h / 8, w / 8)
        # Square root of variance = standard deviation
        stdev = var.sqrt()  # (b, 4, h / 8, w / 8)

        # Given a gaussian distribution, we can convert to another
        # distribution with a given mean and standard deviation
        # through the following transformation:

        # Z = N(0, 1) -> X = N(mean, variance)
        # X = mean + standard deviation * Z
        x = mean + stdev * noise  # (b, 4, h / 8, w / 8)

        # Scale the output by a constant (from original stable diffusion repository)
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215

        # (b, 4, h / 8, w / 8)
        return x
