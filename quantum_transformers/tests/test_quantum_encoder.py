# Test the EncoderBlock class
import torch
from models import EncoderBlock


def test_encoder_block():
    # Define parameters
    embed_len = 64
    num_heads = 8
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    mask = None

    # Create an instance of EncoderBlock
    model = EncoderBlock(embed_len, num_heads, batch_size, dropout, mask)

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
