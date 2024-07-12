import math
import torch
import torch.nn as nn

def scaled_dot_product_attention(queries, keys, values, embed_len, mask=None):
    """
    Perform scaled dot-product attention.

    Parameters:
    queries (Tensor): Query tensor.
    keys (Tensor): Key tensor.
    values (Tensor): Value tensor.
    embed_len (int): Embedding length (dimension of keys and queries).
    mask (Tensor, optional): Mask tensor for masking certain positions.

    Returns:
    Tensor: The result of the attention mechanism applied to the values.
    """
    dk = embed_len  # dimension of keys and queries
    # Compute the compatibility scores
    compatibility = torch.matmul(queries, keys.transpose(-2, -1))
    compatibility = compatibility / math.sqrt(dk)

    # Apply the mask if provided
    if mask is not None:
        compatibility = compatibility.masked_fill(mask == 0, -1e9)

    # Apply softmax on the last dimension
    softmax = nn.Softmax(dim=-1)
    compatibility = softmax(compatibility)

    return torch.matmul(compatibility, values)

# Example usage
if __name__ == "__main__":
    # Sample input tensors
    batch_size = 32
    seq_len = 50
    embed_len = 64

    queries = torch.randn(batch_size, seq_len, embed_len)
    keys = torch.randn(batch_size, seq_len, embed_len)
    values = torch.randn(batch_size, seq_len, embed_len)
    mask = torch.ones(batch_size, seq_len, seq_len)  # Example mask

    # Perform scaled dot-product attention
    result = scaled_dot_product_attention(queries, keys, values, embed_len, mask)
    print(result.shape)  # Should output torch.Size([32, 50, 64])
