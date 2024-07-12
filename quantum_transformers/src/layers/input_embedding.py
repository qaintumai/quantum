import torch
import torch.nn as nn

def input_embedding(input, input_vocab_size, embed_len, dropout=0.1, device='cpu'):
    """
    Classical data embedding function that performs embedding and positional encoding.

    Parameters:
    input (Tensor): Input tensor to be embedded.
    input_vocab_size (int): Vocabulary size for embedding.
    embed_len (int): Length of the embedding vector.
    dropout (float): Dropout rate.
    device (str): Device to run the embedding on (default is 'cpu').

    Returns:
    Tensor: The embedded and positionally encoded tensor.
    """
    # Define the embedding layers and dropout layer
    first_embedding_layer = nn.Embedding(input_vocab_size, embed_len).to(device)
    second_embedding_layer = nn.Embedding(input_vocab_size, embed_len).to(device)
    dropout_layer = nn.Dropout(p=dropout)

    # Perform the embedding and positional encoding
    first_embedding = first_embedding_layer(input).to(device)
    batch_size, seq_len = input.shape

    positions_vector = torch.arange(0, seq_len).expand(batch_size, seq_len).to(device)
    positional_encoding = second_embedding_layer(positions_vector).to(device)

    return dropout_layer(first_embedding + positional_encoding)

# Example usage
if __name__ == "__main__":
    # Sample input tensor
    input_vocab_size = 100
    embed_len = 64
    input_tensor = torch.randint(0, input_vocab_size, (32, 50))  # Batch size of 32, sequence length of 50

    # Perform classical data embedding
    embedded_tensor = classical_data_embedding(input_tensor, input_vocab_size, embed_len, dropout=0.1, device='cpu')
    print(embedded_tensor.shape)  # Should output torch.Size([32, 50, 64])
