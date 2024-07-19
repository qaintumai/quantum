# Test the DecoderBlock class
import torch
from models import DecoderBlock


def test_decoder_block():
    # Define parameters
    embed_len = 64
    num_heads = 8
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    mask = None

    # Create an instance of DecoderBlock
    model = DecoderBlock(embed_len, num_heads, batch_size, dropout, mask)

    # Create dummy input tensors
    target = torch.rand(batch_size, seq_len, embed_len)
    encoder_output = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(target, encoder_output)

    # Check the output shape
    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(
        output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape
