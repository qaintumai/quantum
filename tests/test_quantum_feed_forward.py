# Test the FeedForwardBlock class
import torch
from models import FeedForwardBlock


def test_feed_forward_block():
    # Define parameters
    embed_len = 64
    seq_len = 10
    batch_size = 32
    dropout = 0.1

    # Create an instance of FeedForwardBlock
    model = FeedForwardBlock(embed_len, dropout)

    # Create a dummy input tensor
    dummy_input = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(dummy_input)

    # Check the output shape
    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(
        output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape
