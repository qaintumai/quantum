import torch
import torch.nn as nn

"""
Usage:
To use the InputEmbedding class, import it as follows:
    from layers.input_embedding import InputEmbedding

Example:
    embedding_layer = InputEmbedding(input_vocab_size=10000, embed_len=128)
    output = embedding_layer(input_tensor)
"""

class InputEmbedding(nn.Module):
    """
    A class used to generate embeddings for input data and add positional encodings.
    """

    def __init__(self, input_vocab_size, embed_len, dropout=0.1, device='cpu'):
        """
        Initializes the InputEmbedding class with the given parameters.

        Parameters:
        - input_vocab_size (int): Size of the input vocabulary.
        - embed_len (int): Length of the embedding vector.
        - dropout (float, optional): Dropout rate for regularization. Default is 0.1.
        - device (str, optional): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        
        super(InputEmbedding, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_len = embed_len
        self.dropout = dropout
        self.device = device

        # Define the embedding layers and dropout layer
        self.firstEmbedding = nn.Embedding(
            self.input_vocab_size, self.embed_len).to(self.device)
        self.secondEmbedding = nn.Embedding(
            self.input_vocab_size, self.embed_len).to(self.device)
        self.dropoutLayer = nn.Dropout(p=self.dropout)

    
    def forward(self, input):
        """
        Computes the embeddings and positional encodings for the input data.

        Parameters:
        - input (torch.Tensor): Input tensor containing the data to be embedded.

        Returns:
        - torch.Tensor: Tensor containing the combined token embeddings and positional encodings with dropout applied.
        """
        # Compute the token embeddings
        first_embedding = self.firstEmbedding(input).to(self.device)
        batch_size, seq_len = input.shape

        positions_vector = torch.arange(0, seq_len).expand(
            batch_size, seq_len).to(self.device)
        positional_encoding = self.secondEmbedding(
            positions_vector).to(self.device)

        return self.dropoutLayer(first_embedding + positional_encoding)
