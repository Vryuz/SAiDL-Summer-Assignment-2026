"""
conv_hybrid.py — Convolution + Attention Hybrid Blocks

Implements two hybrid designs as required by the SAiDL Core ML assignment Part 4:
  1. ConvBeforeAttentionBlock: Conv1D applied before each attention layer to
     capture local n-gram context before global attention.
  2. GatedConvFFNBlock: The standard FFN is replaced with a Gated Depthwise
     Separable Convolutional FFN (inspired by Conformer / ConvGLU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import StandardAttention


class CausalConv1d(nn.Module):
    """
    A causal 1D convolution that ensures token at position t
    only sees positions <= t (no future leakage).
    Implemented via left-padding of (kernel_size - 1) zeros.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, groups=groups
        )

    def forward(self, x):
        # x: (B, T, C) -> need (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)  # back to (B, T, C)


class ConvBeforeAttentionBlock(nn.Module):
    """
    Design 1: Conv1D layer inserted before each attention block.
    
    Architecture:
        x -> LayerNorm -> CausalConv1d (local n-gram features) -> residual
          -> LayerNorm -> StandardAttention (global context)     -> residual
          -> LayerNorm -> FeedForward                            -> residual
    
    The conv layer learns local dependencies cheaply before the attention
    layer handles long-range relationships.
    """
    def __init__(self, embed_dim, num_heads, kernel_size=7, pos_enc="sinusoidal"):
        super().__init__()
        self.ln_conv = nn.LayerNorm(embed_dim)
        self.conv = CausalConv1d(embed_dim, embed_dim, kernel_size=kernel_size)
        self.conv_act = nn.GELU()
        self.conv_proj = nn.Linear(embed_dim, embed_dim)

        self.ln_attn = nn.LayerNorm(embed_dim)
        self.attn = StandardAttention(embed_dim, num_heads, pos_enc=pos_enc)

        self.ln_ff = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, mask=None, rel_embeddings=None):
        # 1. Causal Conv sub-layer
        x = x + self.conv_proj(self.conv_act(self.conv(self.ln_conv(x))))
        # 2. Attention sub-layer
        x = x + self.attn(self.ln_attn(x), mask=mask, rel_embeddings=rel_embeddings)
        # 3. FFN sub-layer
        x = x + self.ff(self.ln_ff(x))
        return x


class GatedConvFFNBlock(nn.Module):
    """
    Design 2: Gated Depthwise Separable Convolutional FFN.
    
    Architecture:
        x -> LayerNorm -> StandardAttention -> residual
          -> LayerNorm -> GatedConvFFN      -> residual
    
    The standard FFN (2 linear layers) is replaced with a gated 1D conv network:
        FFN(x) = (Linear_up(x) * sigmoid(Linear_gate(x))) -> DepthwiseConv1d -> Linear_down
    
    This is inspired by the Conformer architecture and gated linear units (GLU),
    combining the channel-mixing of FFN with local convolutional context.
    """
    def __init__(self, embed_dim, num_heads, kernel_size=31, pos_enc="sinusoidal"):
        super().__init__()
        hidden_dim = embed_dim * 4

        self.ln_attn = nn.LayerNorm(embed_dim)
        self.attn = StandardAttention(embed_dim, num_heads, pos_enc=pos_enc)

        self.ln_ff = nn.LayerNorm(embed_dim)
        # Gated linear unit: project up to 2x, split into value and gate
        self.linear_up = nn.Linear(embed_dim, hidden_dim * 2)
        # Depthwise separable conv on the gated features
        self.dw_conv = CausalConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim)
        self.pw_conv = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, mask=None, rel_embeddings=None):
        # 1. Attention sub-layer
        x = x + self.attn(self.ln_attn(x), mask=mask, rel_embeddings=rel_embeddings)
        # 2. Gated Conv FFN sub-layer
        h = self.linear_up(self.ln_ff(x))
        # Split into value and gate
        v, g = h.chunk(2, dim=-1)
        # Gate with sigmoid (GLU)
        h = v * torch.sigmoid(g)
        # Depthwise conv for local mixing
        h = self.dw_conv(h)
        x = x + self.pw_conv(h)
        return x
