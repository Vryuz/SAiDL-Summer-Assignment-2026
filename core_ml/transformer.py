import math
import torch
import torch.nn as nn
from attention import StandardAttention, MultiQueryAttention, SlidingWindowAttention, LinearAttention
from conv_hybrid import ConvBeforeAttentionBlock, GatedConvFFNBlock
from rope import RotaryPositionalEncoding
from alibi import AlibiPositionalEncoding
from relative_pe import RelativePositionalEncoding

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=8192):
        super().__init__()
        # Create a matrix of shape (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (B, T, C)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_type="standard", pos_enc="sinusoidal"):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Pass pos_enc to the attention mechanism
        if attn_type == "standard":
            self.attn = StandardAttention(embed_dim, num_heads, pos_enc=pos_enc)
        elif attn_type == "mqa":
            self.attn = MultiQueryAttention(embed_dim, num_heads, pos_enc=pos_enc)
        elif attn_type == "sliding_window":
            self.attn = SlidingWindowAttention(embed_dim, num_heads, pos_enc=pos_enc, window_size=64)
        elif attn_type == "linear":
            self.attn = LinearAttention(embed_dim, num_heads, pos_enc=pos_enc)
        elif attn_type == "conv_before":
            self.attn = ConvBeforeAttentionBlock(embed_dim, num_heads, kernel_size=7, pos_enc=pos_enc)
        elif attn_type == "gated_conv_ffn":
            self.attn = GatedConvFFNBlock(embed_dim, num_heads, kernel_size=31, pos_enc=pos_enc)
        else:
            raise NotImplementedError(f"Attention type {attn_type} not implemented yet.")
            
        self.ff = FeedForward(embed_dim, embed_dim * 4)

    def forward(self, x, mask=None, rel_embeddings=None):
        # Pre-LN architecture (LayerNorm before attention/ffn)
        x = x + self.attn(self.ln1(x), mask=mask, rel_embeddings=rel_embeddings)
        x = x + self.ff(self.ln2(x))
        return x

class SequenceModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, attn_type="standard", pos_enc="sinusoidal"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = pos_enc
        self.num_heads = num_heads
        
        if pos_enc == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(embed_dim)
        elif pos_enc == "rope":
            self.pos_emb = RotaryPositionalEncoding(embed_dim)
        elif pos_enc == "alibi":
            self.pos_emb = AlibiPositionalEncoding(embed_dim)
        elif pos_enc == "relative":
            self.pos_emb = RelativePositionalEncoding(embed_dim // num_heads, max_relative_position=128)
        else:
            raise NotImplementedError(f"Positional encoding {pos_enc} not implemented yet.")
            
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, attn_type=attn_type, pos_enc=pos_enc) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.size()
        
        # Embed tokens
        x = self.token_emb(x)
        
        # Apply positional encoding
        if self.pos_enc == "relative":
            rel_embeddings = self.pos_emb(T, x.device)
        else:
            rel_embeddings = None
            x = self.pos_emb(x)
        
        # Create causal mask (lower triangular)
        # (T, T) -> (1, 1, T, T) so it broadcasts over batch and heads
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        # If ALiBi, combine causal mask and linear bias
        if self.pos_enc == "alibi":
            from alibi import get_alibi_bias
            # Convert binary causal mask to additive mask (valid positions = 0.0, masked = -inf)
            additive_mask = torch.zeros_like(mask, dtype=x.dtype)
            additive_mask = additive_mask.masked_fill(mask == 0, float('-inf'))
            
            # Generate ALiBi bias
            alibi_bias = get_alibi_bias(self.num_heads, T, device=x.device, dtype=x.dtype)
            
            # Combine them
            mask = additive_mask + alibi_bias
        
        for block in self.blocks:
            x = block(x, mask=mask, rel_embeddings=rel_embeddings)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
