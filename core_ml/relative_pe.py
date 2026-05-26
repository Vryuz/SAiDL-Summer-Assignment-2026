import torch
import torch.nn as nn

class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_relative_position):
        """
        Initializes the Relative Positional Encoding module (Shaw et al.).
        
        Args:
            embed_dim: The dimension of the embeddings (head_dim).
            max_relative_position: The maximum relative distance to consider.
        """
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embed_dim = embed_dim
        
        # We need embeddings for relative distances from -max_relative_position to max_relative_position
        # Total number of embeddings = 2 * max_relative_position + 1
        vocab_size = 2 * max_relative_position + 1
        self.relative_embeddings = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, seq_len, device):
        """
        Generates the relative positional embeddings for a sequence of length seq_len.
        
        Args:
            seq_len: The length of the sequence.
            device: The device to place the tensors on.
            
        Returns:
            rel_embeddings: Tensor of shape (seq_len, seq_len, embed_dim)
        """
        # Create a range of positions
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        
        # Calculate relative distances (query_pos - key_pos)
        # Using broadcasting: shape will be (seq_len, seq_len)
        rel_distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Clip distances to the range [-max_relative_position, max_relative_position]
        rel_distances = torch.clamp(rel_distances, -self.max_relative_position, self.max_relative_position)
        
        # Shift the distances to be non-negative indices (0 to 2*max_relative_position)
        rel_indices = rel_distances + self.max_relative_position
        
        # Get the embeddings: shape (seq_len, seq_len, embed_dim)
        rel_embeddings = self.relative_embeddings(rel_indices)
        
        return rel_embeddings

