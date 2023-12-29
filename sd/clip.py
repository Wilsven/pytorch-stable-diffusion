import torch
from torch import nn

from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """Converts the text prompt into embeddings with positional information."""

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens: torch.LongTensor) -> torch.LongTensor:
        # tokens.shape: (b, seq_len)
        tokens = self.token_embedding(tokens)  # (b, seq_len) -> (b, seq_len, n_embed)
        tokens += (
            self.position_embedding
        )  # (b, seq_len, n_embed) -> (b, seq_len, n_embed)

        # (b, seq_len, n_embed)
        return tokens


class CLIPLayer(nn.Module):
    """Performs the self-attention mechanism on the prompt embeddings."""

    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, seq_len, n_embed)

        residual = x

        ## Self Attention
        x = self.layernorm_1(x)
        x = self.attention(x, mask=True)
        x += residual  # skip connection

        residual = x

        ## Feed Forward
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # No particular reason why this activation function was used.
        # Simply, it was found to work well in practice.
        x *= torch.sigmoid(1.702 * x)  # QuickGELU activation function
        x = self.linear_2(x)
        x += residual  # skip connection

        # (b, seq_len, n_embed)
        return x


class CLIP(nn.Module):
    """The final Contrastive Language-Image Pre-training (CLIP) model."""

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for _ in range(12)])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens.shape: (b, seq_len)
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)  # (b, seq_len) -> (b, seq_len, n_embed)

        for layer in self.layers:
            state = layer(state)

        # (b, seq_len, n_embed) -> (b, seq_len, n_embed)
        output = self.layernorm(state)

        # (b, seq_len, n_embed)
        return output
