import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardAttention(nn.Module):
    """
    Standard multi-head self-attention with support for relative positional
    encodings, ALiBi masks, and standard additive masking.
    """
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

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA). Shares a single Key and Value projection across 
    all Query heads to reduce memory bandwidth during generation.
    """
    def __init__(self, embed_dim, num_heads, pos_enc="sinusoidal"):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pos_enc = pos_enc

        # Query projection (per head)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key and Value projections (shared across all heads, so output dim is head_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        
        # Output projection
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, rel_embeddings=None):
        B, T, C = x.size()
        
        # 1. Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape 
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2) # (B, 1, T, head_dim)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2) # (B, 1, T, head_dim)
        
        # Apply RoPE if specified
        if self.pos_enc == "rope":
            from rope import get_rotary_emb, apply_rotary_emb
            cos, sin = get_rotary_emb(self.head_dim, T, x.device, x.dtype)
            q, k = apply_rotary_emb(q, k, cos, sin)
            
        # 3. Scaled Dot-Product Attention
        # To compute scores, we broadcast k to all heads
        # k.transpose(-2, -1): (B, 1, head_dim, T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Add relative positional embeddings to scores
        if self.pos_enc == "relative" and rel_embeddings is not None:
            rel_scores = torch.einsum('bhid,ijd->bhij', q, rel_embeddings)
            scores = scores + rel_scores
            
        scores = scores / math.sqrt(self.head_dim)
        
        # Apply causal mask (if provided)
        if mask is not None:
            if torch.is_floating_point(mask):
                scores = scores + mask  
            else:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 4. Softmax and weighted sum of values
        attn_weights = F.softmax(scores, dim=-1)
        
        # v is (B, 1, T, head_dim). Broadcast across heads during matmul
        out = torch.matmul(attn_weights, v) # (B, num_heads, T, head_dim)
        
        # 5. Concatenate heads back together
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 6. Final linear projection
        return self.o_proj(out)
class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention. Restricts the receptive field of each token to a 
    fixed window of previous tokens, reducing computational complexity.
    """
    def __init__(self, embed_dim, num_heads, pos_enc="sinusoidal", window_size=64):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pos_enc = pos_enc
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, rel_embeddings=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.pos_enc == "rope":
            from rope import get_rotary_emb, apply_rotary_emb
            cos, sin = get_rotary_emb(self.head_dim, T, x.device, x.dtype)
            q, k = apply_rotary_emb(q, k, cos, sin)
            
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if self.pos_enc == "relative" and rel_embeddings is not None:
            rel_scores = torch.einsum('bhid,ijd->bhij', q, rel_embeddings)
            scores = scores + rel_scores
            
        scores = scores / math.sqrt(self.head_dim)
        
        # Apply sliding window causal mask
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        window = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-(self.window_size - 1))
        sw_mask = causal & window
        
        sw_mask_float = torch.zeros(T, T, device=x.device)
        sw_mask_float.masked_fill_(~sw_mask, float('-inf'))
        scores = scores + sw_mask_float
        
        if mask is not None:
            if torch.is_floating_point(mask):
                scores = scores + mask  
            else:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class LinearAttention(nn.Module):
    """
    Linear Attention. Uses a non-negative feature map to approximate standard 
    attention with linear time and memory complexity.
    """
    def __init__(self, embed_dim, num_heads, pos_enc="none"):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pos_enc = pos_enc # Linear attention typically doesn't use relative/alibi effectively in O(T), so we only support RoPE/Sinusoidal

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def feature_map(self, x):
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x, mask=None, rel_embeddings=None):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.pos_enc == "rope":
            from rope import get_rotary_emb, apply_rotary_emb
            cos, sin = get_rotary_emb(self.head_dim, T, x.device, x.dtype)
            q, k = apply_rotary_emb(q, k, cos, sin)
            
        # Apply feature map to make keys and queries strictly positive
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Causal linear attention O(T):
        # kv: (B, num_heads, T, head_dim, head_dim)
        kv = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        
        # Cumulative sum over time
        kv_cumsum = torch.cumsum(kv, dim=2)
        k_cumsum = torch.cumsum(k, dim=2)
        
        # Compute numerator
        num = torch.matmul(q.unsqueeze(-2), kv_cumsum).squeeze(-2)
        
        # Compute denominator
        den = torch.sum(q * k_cumsum, dim=-1, keepdim=True) + 1e-6
        
        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
