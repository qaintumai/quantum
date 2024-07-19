# Test the ScaledDotProduct class
import torch
from models import ScaledDotProduct


# Test the ScaledDotProduct class
def test_scaled_dot_product():
    # Define parameters
    embed_len = 64
    seq_len = 10
    batch_size = 32

    # Create an instance of ScaledDotProduct
    model = ScaledDotProduct(embed_len)

    # Create dummy input tensors
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(queries, keys, values)

    # Check the output shape
    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(
        output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")
    return output.shape
