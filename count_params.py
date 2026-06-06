import torch
import sys
import os
sys.path.append(os.path.abspath('core_ml'))
from transformer import TransformerLanguageModel
from attention import StandardAttention

# Assuming a standard vocab size for WikiText-2 word level (e.g., 33278) or whatever they use
# Let's check dataset.py if it exists, or just pass a dummy vocab size
vocab_size = 33278 
embed_dim = 256
num_layers = 4
num_heads = 4

model = TransformerLanguageModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    attn_type='standard',
    pos_enc='rope'
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Core ML Transformer (embed=256, layers=4): {total_params:,} total parameters")

# The DiT-B/8 model is standard.
# DiT-B (Base) has ~130M parameters.
