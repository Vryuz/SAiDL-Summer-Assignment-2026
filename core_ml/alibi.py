import math
import torch
import torch.nn as nn

class AlibiPositionalEncoding(nn.Module):
    """
    ALiBi Positional Encoding wrapper module.
    
    ALiBi does not modify token embeddings directly; it is applied as a relative bias
    to the attention matrix. Therefore, this module acts as a direct pass-through,
    preserving the clean pre-LayerNorm sequence model architecture interface:
    "Every positional encoding must be a nn.Module with forward(x) returning x + encoding"
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        return x

def get_slopes(num_heads):
    """
    Computes the geometric slope values for the ALiBi bias across attention heads.
    
    For a model with H heads, the slope for the h-th head is:
    m_h = 2^(-8 * h / H) if H is a power of 2.
    If H is not a power of 2, it interpolates from the closest power of 2.
    """
    def get_slopes_power_of_2(n):
        start = 2**(-8/n)
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    else:
        closest_power_of_2 = 2**int(math.log2(num_heads))
        slopes_base = get_slopes_power_of_2(closest_power_of_2)
        slopes_extra = get_slopes_power_of_2(closest_power_of_2 * 2)[1::2]
        return torch.tensor((slopes_base + slopes_extra)[:num_heads])

def get_alibi_bias(num_heads, seq_len, device, dtype):
    """
    Generates the additive linear bias matrix for ALiBi.
    
    For query index i and key index j, the bias penalizes their interaction by:
    -m * |i - j|
    
    Shape of output: (1, num_heads, seq_len, seq_len)
    """
    slopes = get_slopes(num_heads).to(device=device, dtype=dtype) # (num_heads,)
    
    # Generate relative position distance matrix |i - j|
    r = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1) # (seq_len, 1)
    c = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0) # (1, seq_len)
    distance_matrix = torch.abs(r - c) # (seq_len, seq_len)
    
    # Reshape to (1, num_heads, seq_len, seq_len) for broadcasting
    bias = -slopes.view(1, num_heads, 1, 1) * distance_matrix.unsqueeze(0).unsqueeze(0)
    
    return bias
