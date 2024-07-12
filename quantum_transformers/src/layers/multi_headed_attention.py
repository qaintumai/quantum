import torch
import torch.nn as nn
from .scaled_dot_product import scaled_dot_product_attention

def multi_headed_attention(queries, keys, values, num_heads, embed_len, batch_size, mask=None):
    """
    Perform multi-headed attention.

    Parameters:
    queries (Tensor): Query tensor.
    keys (Tensor): Key tensor.
    values (Tensor): Value tensor.
    num_heads (int): Number of attention heads.
    embed_len (int): Embedding length (dimension of keys and queries).
    batch_size (int): Batch size.
    mask (Tensor, optional): Mask tensor for masking certain positions.

    Returns:
    Tensor: The result of the multi-headed attention mechanism applied to the values.
    """
    head_length = embed_len // num_heads

    q_linear = nn.Linear(embed_len, embed_len)
    k_linear = nn.Linear(embed_len, embed_len)
    v_linear = nn.Linear(embed_len, embed_len)
    output_linear = nn.Linear(embed_len, embed_len)

    queries = q_linear(queries).reshape(batch_size, -1, num_heads, head_length)
    queries = queries.transpose(1, 2)

    keys = k_linear(keys).reshape(batch_size, -1, num_heads, head_length)
    keys = keys.transpose(1, 2)

    values = v_linear(values).reshape(batch_size, -1, num_heads, head_length)
    values = values.transpose(1, 2)

    sdp_output = scaled_dot_product_attention(queries, keys, values, head_length, mask)
    sdp_output = sdp_output.transpose(1, 2).reshape(batch_size, -1, num_heads * head_length)

    return output_linear(sdp_output)

# Example usage
if __name__ == "__main__":
    # Sample input tensors
    batch_size = 32
    seq_len = 50
    embed_len = 64
    num_heads = 8

    queries = torch.randn(batch_size, seq_len, embed_len)
    keys = torch.randn(batch_size, seq_len, embed_len)
    values = torch.randn(batch_size, seq_len, embed_len)
    mask = torch.ones(batch_size, seq_len, seq_len)  # Example mask

    # Perform multi-headed attention
    result = multi_headed_attention(queries, keys, values, num_heads, embed_len, batch_size, mask)
    print(result.shape)  # Should output torch.Size([32, 50, 64])
