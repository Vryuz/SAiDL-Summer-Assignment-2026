import math
import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) wrapper module.
    
    Conforms to the assignment's positional encoding interface:
    "Every positional encoding must be a nn.Module with forward(x) returning x + encoding"
    
    Since RoPE is mathematically applied to the Query (Q) and Key (K) representations 
    inside the self-attention layer itself (rather than being added to the input word 
    embeddings), this module acts as a zero-addition no-op at the token level,
    while passing the RoPE configuration flag down to the attention layer.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # RoPE is applied during attention calculation, not on embeddings
        return x

def rotate_half(x):
    """
    Splits the last dimension of the tensor into two halves,
    negates the second half, and concatenates them.
    
    Used to implement complex number multiplication in a vectorized way:
    (a + ib) * (cos + isin) = (a*cos - b*sin) + i(b*cos + a*sin)
    
    Args:
        x: Tensor of shape (B, num_heads, T, head_dim)
    Returns:
        Tensor of shape (B, num_heads, T, head_dim)
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def get_rotary_emb(head_dim, seq_len, device, dtype):
    """
    Generates dynamic cosine and sine rotation matrices for a given sequence length.
    Dynamically computing this enables extrapolation to arbitrary context lengths
    (e.g., evaluating on 1024 or 2048 after training on 512) and saves VRAM.
    
    Args:
        head_dim: Dimension of each attention head.
        seq_len: Length of the current sequence.
        device: Device to place the tensors on.
        dtype: Data type of the tensors.
        
    Returns:
        cos: Cosine frequencies tensor of shape (1, 1, seq_len, head_dim)
        sin: Sine frequencies tensor of shape (1, 1, seq_len, head_dim)
    """
    # 1. Compute theta values: theta_i = 10000^(-2i / d) for i in [0, 2, ..., d-2]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    
    # 2. Compute position indexes: t = [0, 1, ..., seq_len-1]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # 3. Compute outer product: freqs[m, i] = m * theta_i of shape (seq_len, head_dim // 2)
    freqs = torch.outer(t, inv_freq)
    
    # 4. Duplicate frequencies for both real and imaginary components: shape (seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
    
    # 5. Compute cosine and sine, then unsqueeze to (1, 1, seq_len, head_dim) for broadcasting
    cos = emb.cos().unsqueeze(0).unsqueeze(1)
    sin = emb.sin().unsqueeze(0).unsqueeze(1)
    
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    """
    Applies RoPE rotation to query and key tensors.
    
    Args:
        q: Query tensor of shape (B, num_heads, T, head_dim)
        k: Key tensor of shape (B, num_heads, T, head_dim)
        cos: Cosine frequencies tensor of shape (1, 1, T, head_dim)
        sin: Sine frequencies tensor of shape (1, 1, T, head_dim)
        
    Returns:
        q_rot: Rotated query tensor of shape (B, num_heads, T, head_dim)
        k_rot: Rotated key tensor of shape (B, num_heads, T, head_dim)
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
