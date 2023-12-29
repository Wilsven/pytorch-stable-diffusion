import torch
from torch import nn
from torch.nn import functional as F

from attention import CrossAttention, SelfAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1, 320) -> (1, 1280)
        return self.linear_2(self.silu(self.linear_1(x)))


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_embed: int = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear_time = nn.Linear(n_embed, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        self.silu = nn.SiLU()

        # To ensure the output_channels and input_channels
        # are equal before we can add the skip connection
        self.residual_layer = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature.shape: (b, in_channels, h, w)
        # time.shape: (1, 1280)

        residual = feature

        # (b, in_channels, h, w) -> (b, out_channels, h, w)
        feature = self.conv_feature(self.silu(self.groupnorm_feature(feature)))
        # (1, 1280) -> (1, out_channels)
        time = self.linear_time(self.silu(time))

        # (b, out_channels, h, w)
        x = feature + time.unsqueeze(-1).unsqueeze(-1)
        # (b, out_channels, h, w) -> (b, out_channels, h, w)
        x = self.conv_merged(self.silu(self.groupnorm_merged(x)))

        # (b, out_channels, h, w)
        return self.residual_layer(residual) + x


class UNetAttentionBlock(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_context: int = 768):
        super().__init__()
        channels = n_heads * d_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_heads, channels, d_context, in_proj_bias=False
        )

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)

        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, c, h, w)
        # context.shape: (b, seq_len, d_embed)
        b, c, h, w = x.shape

        final_residual = x

        # (b, c, h, w) -> (b, c, h, w)
        x = self.conv_input(self.groupnorm(x))
        # (b, c, h, w) -> (b, c, h * w)
        x = x.view(b, c, h * w)
        # (b, c, h * w) -> (b, h * w, c)
        x = x.transpose(-1, -2)

        ## Normalization + Self Attention with skip connection
        residual = x

        # (b, h * w, c) -> (b, h * w, c)
        x = self.attention_1(self.layernorm_1(x))
        x += residual

        ## Normalization + Cross Attention with skip connection
        residual = x

        # (b, h * w, c) -> (b, h * w, c)
        x = self.attention_2(self.layernorm_2(x), context)
        x += residual

        ## Normalizaion + FFN with GeGLU and skip connection
        residual = x

        # GeGLU as implemented in the original code:
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (b, h * w, c) -> (b, h * w, 4 * c * 2)
        x = self.linear_geglu_1(self.layernorm_3(x))
        # (b, h * w, 4 * c * 2) -> 2 x (b, h * w, 4 * c)
        x, gate = x.chunk(2, dim=-1)
        # Element-wise product
        x *= self.gelu(gate)

        # (b, h * w, 4 * c) -> (b, h * w, c)
        x = self.linear_geglu_2(x)
        x += residual

        # (b, h * w, c) -> (b, c, h * w)
        x = x.transpose(-1, -2)
        # (b, c, h * w) -> (b, c, h, w)
        x = x.view(b, c, h, w)

        # (b, c, h, w)
        return final_residual + self.conv_output(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b, c, h, w) -> (b, c, h * 2, w * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(
                    x, context
                )  # computes the cross attention between latent and prompt (context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                # (b, 4, h / 8, w / 8) -> (b, 320, h / 8, w / 8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                # (b, 320, h / 8, w / 8) -> (b, 320, h / 8, w / 8)
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                # (b, 320, h / 8, w / 8) -> (b, 320, h / 8, w / 8)
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                # (b, 320, h / 8, w / 8) -> (b, 320, h / 16, w / 16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                # (b, 320, h / 16, w / 16) -> (b, 640, h / 16, w / 16)
                SwitchSequential(
                    UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)
                ),
                # (b, 640, h / 16, w / 16) -> (b, 640, h / 16, w / 16)
                SwitchSequential(
                    UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)
                ),
                # (b, 640, h / 16, w / 16) -> (b, 640, h / 32, w / 32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                # (b, 640, h / 32, w / 32) -> (b, 1280, h / 32, w / 32)
                SwitchSequential(
                    UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)
                ),
                # (b, 1280, h / 32, w / 32) -> (b, 1280, h / 32, w / 32)
                SwitchSequential(
                    UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)
                ),
                # (b, 1280, h / 32, w / 32) -> (b, 1280, h / 64, w / 64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                # (b, 1280, h / 64, w / 64) -> (b, 1280, h / 64, w / 64)
                SwitchSequential(UNetResidualBlock(1280, 1280)),
                # (b, 1280, h / 64, w / 64) -> (b, 1280, h / 64, w / 64)
                SwitchSequential(UNetResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # (b, 2560, h / 64, w / 64) -> (b, 1280, h / 64, w / 64)
                SwitchSequential(UNetResidualBlock(2560, 1280)),
                # (b, 2560, h / 64, w / 64) -> (b, 1280, h / 64, w / 64)
                SwitchSequential(UNetResidualBlock(2560, 1280)),
                # (b, 2560, h / 64, w / 64) -> (b, 1280, h / 64, w / 64) -> (b, 1280, h / 32, w / 32)
                SwitchSequential(UNetResidualBlock(2560, 1280), Upsample(1280)),
                # (b, 2560, h / 32, w / 32) -> # (b, 1280, h / 32, w / 32)
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                # (b, 2560, h / 32, w / 32) -> (b, 1280, h / 32, w / 32)
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                # (b, 1920, h / 32, w / 32) -> (b, 1280, h / 32, w / 32) -> (b, 1280, h / 16, w / 16)
                SwitchSequential(
                    UNetResidualBlock(1920, 1280),
                    UNetAttentionBlock(8, 160),
                    Upsample(1280),
                ),
                # (b, 1920, h / 16, w / 16) -> (b, 640, h / 16, w / 16)
                SwitchSequential(
                    UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)
                ),
                # (b, 1280, h / 16, w / 16) -> (b, 640, h / 16, w / 16)
                SwitchSequential(
                    UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)
                ),
                # (b, 960, h / 16, w / 16) -> (b, 640, h / 16, w / 16) -> (b, 640, h / 8, w / 8)
                SwitchSequential(
                    UNetResidualBlock(960, 640),
                    UNetAttentionBlock(8, 80),
                    Upsample(640),
                ),
                # (b, 960, h / 8, w / 8) -> (b, 320, h / 8, w / 8)
                SwitchSequential(
                    UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)
                ),
                # (b, 640, h / 8, w / 8) -> (b, 320, h / 8, w / 8)
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
                # (b, 640, h / 8, w / 8) -> (b, 320, h / 8, w / 8)
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # x.shape: (b, 4, h / 8, w / 8)
        # context.shape: (b, seq_len, n_embed)
        # time.shape: (1, 1280)

        skip_connections = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoders:
            # Since we always concat with the skip connection of the encoder,
            # the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, context, time)

        # (b, 320, h / 8, w / 8)
        return x


class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, 320, h / 8, w / 8)

        # (b, 320, h / 8, w / 8) -> (b, 4, h / 8, w / 8)
        x = self.conv(self.silu(self.groupnorm(x)))

        # (b, 4, h / 8, w / 8)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutputLayer(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # latent.shape: (b, 4, h / 8, w / 8)
        # context.shape: (b, seq_len, n_embed)
        # time.shape: (1, 320)

        time_embed = self.time_embedding(time)  # (1, 320) -> (1, 1280)
        output = self.unet(
            latent, context, time_embed
        )  # (b, 4, h / 8, w / 8) -> (b, 320, h / 8, w / 8)
        output = self.final(output)  # (b, 320, h / 8, w / 8) -> (b, 4, h / 8, w / 8)

        # (b, 4, h / 8, w / 8)
        return output
