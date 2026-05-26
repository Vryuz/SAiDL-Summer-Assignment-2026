import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, pos_enc="sinusoidal"):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pos_enc = pos_enc

        # Key, Query, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, rel_embeddings=None):
        B, T, C = x.size() # Batch, Time (Sequence Length), Channels (Embed Dim)
        
        # 1. Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape for multi-head attention: (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if specified
        if self.pos_enc == "rope":
            from rope import get_rotary_emb, apply_rotary_emb
            cos, sin = get_rotary_emb(self.head_dim, T, x.device, x.dtype)
            q, k = apply_rotary_emb(q, k, cos, sin)
            
        # 3. Scaled Dot-Product Attention
        # q: (B, num_heads, T, head_dim)
        # k.transpose(-2, -1): (B, num_heads, head_dim, T)
        # scores: (B, num_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Add relative positional embeddings to scores
        if self.pos_enc == "relative" and rel_embeddings is not None:
            # rel_embeddings: (T, T, head_dim)
            # q: (B, num_heads, T, head_dim)
            rel_scores = torch.einsum('bhid,ijd->bhij', q, rel_embeddings)
            scores = scores + rel_scores
            
        scores = scores / math.sqrt(self.head_dim)
        
        # Apply causal mask (if provided)
        if mask is not None:
            if torch.is_floating_point(mask):
                scores = scores + mask  # Additive floating-point mask (e.g. Causal + ALiBi)
            else:
                scores = scores.masked_fill(mask == 0, float('-inf'))  # Binary/Boolean mask
            
        # 4. Softmax and weighted sum of values
        attn_weights = F.softmax(scores, dim=-1)
        # output: (B, num_heads, T, head_dim)
        out = torch.matmul(attn_weights, v)
        
        # 5. Concatenate heads back together
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 6. Final linear projection
        return self.o_proj(out)
