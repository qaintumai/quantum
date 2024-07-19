import torch
from models import InputEmbedding
from utils.device_configuration import get_device


def test_input_embedding():
    # Define parameters
    input_vocab_size = 100
    embed_len = 64
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    # device = 'cpu'
    device = get_device()

    # Create an instance of InputEmbedding
    model = InputEmbedding(input_vocab_size, embed_len, dropout, device)

    # Create a dummy input tensor with random integers in the range of the vocabulary size
    dummy_input = torch.randint(
        0, input_vocab_size, (batch_size, seq_len)).to(device)

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


if __name__ == '__main__':
    test_input_embedding()
