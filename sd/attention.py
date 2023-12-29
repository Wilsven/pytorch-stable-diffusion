import math

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """Performs the self-attetion mechanism."""

    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        # x.shape: (b, seq_len, d_embed)
        b, seq_len, _ = x.shape

        # Linear projection from input to query, key and value
        # vectors by splitting into 3 across last dimension
        q, k, v = self.in_proj(x).chunk(
            3, dim=-1
        )  # (b, seq_len, d_embed) -> (b, seq_len, d_embed * 3) -> (b, seq_len, d_embed / 3)

        # (b, seq_len, d_embed / 3) -> (b, seq_len, n_heads, d_head) -> (b, n_heads, seq_len, d_head)
        q = q.view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # (b, n_heads, seq_len, d_head) @ (b, n_heads, d_head, seq_len) -> (b, n_heads, seq_len, seq_len)
        attention = q @ k.transpose(-1, -2)

        if mask:
            # Mask the upper triangle (above the principal diagonal) is made up of 1's
            mask = torch.ones_like(attention, dtype=torch.bool).triu(1)
            attention.masked_fill_(mask, -torch.inf)

        attention /= math.sqrt(self.d_head)
        attention = F.softmax(attention, dim=-1)

        # (b, n_heads, seq_len, seq_len) @ (b, n_heads, seq_len, d_head) -> (b, n_heads, seq_len, d_head)
        output = attention @ v
        # (b, n_heads, seq_len, d_head) -> (b, seq_len, n_heads, d_head)
        output = output.transpose(1, 2)
        # (b, seq_len, n_heads, d_head) -> (b, seq_len, n_heads * d_head)
        output = output.reshape(b, seq_len, self.n_heads * self.d_head)
        # (b, seq_len, d_embed) -> (b, seq_len, d_embed)
        output = self.out_proj(output)

        # (b, seq_len, d_embed)
        return output


class CrossAttention(nn.Module):
    """Performs the cross-attention mechanism. Causal mask is provided to prevent look ahead."""

    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        d_context: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_context, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_context, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x.shape (latent): (b, seq_len_q, dim_q) -> (b, seq_len, d_embed)
        # y.shape (context): (b, seq_len_kv, dim_kv) -> (b, 77, 768)

        b, seq_len, _ = x.shape  # (b, seq_len, d_embed)
        interim_shape = (b, -1, self.n_heads, self.d_head)

        ## Multiply query by Wq
        # (b, seq_len, d_embed) -> (b, seq_len, d_embed)
        q = self.q_proj(x)

        ## Multiply key and value by Wk and Wv respectively
        # (b, 77, 768) -> (b, 77, d_embed)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (b, seq_len, d_embed) -> (b, seq_len, n_heads, d_head) -> (b, n_heads, seq_len, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        # (b, 77, d_embed) -> (b, 77, n_heads, d_head) -> (b, n_heads, 77, d_head)
        k = k.view(interim_shape).transpose(1, 2)
        # (b, 77, d_embed) -> (b, 77, n_heads, d_head) -> (b, n_heads, 77, d_head)
        v = v.view(interim_shape).transpose(1, 2)

        # (b, n_heads, seq_len, d_head) @ (b, n_heads, d_head, 77) -> (b, n_heads, seq_len, 77)
        attention = q @ k.transpose(-1, -2)
        attention /= math.sqrt(self.d_head)
        attention = F.softmax(attention, dim=-1)

        # (b, n_heads, seq_len, 77) @ (b, n_heads, 77, d_head) -> (b, n_heads, seq_len, d_head)
        output = attention @ v
        # (b, n_heads, seq_len, d_head) -> (b, seq_len, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()
        # (b, seq_len, n_heads, d_head) -> (b, seq_len, n_heads * d_head)
        output = output.reshape(b, seq_len, self.n_heads * self.d_head)
        # (b, seq_len, d_embed) -> (b, seq_len, d_embed)
        output = self.out_proj(output)

        # (b, seq_len, d_embed)
        return output
