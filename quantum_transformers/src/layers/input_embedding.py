import torch
import torch.nn as nn

# Define the InputEmbedding class


class InputEmbedding(nn.Module):
    def __init__(self, input_vocab_size, embed_len, dropout=0.1, device='cpu'):
        super(InputEmbedding, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_len = embed_len
        self.dropout = dropout
        self.device = device

        self.firstEmbedding = nn.Embedding(
            self.input_vocab_size, self.embed_len).to(self.device)
        self.secondEmbedding = nn.Embedding(
            self.input_vocab_size, self.embed_len).to(self.device)
        self.dropoutLayer = nn.Dropout(p=self.dropout)

    def forward(self, input):
        first_embedding = self.firstEmbedding(input).to(self.device)
        batch_size, seq_len = input.shape

        positions_vector = torch.arange(0, seq_len).expand(
            batch_size, seq_len).to(self.device)
        positional_encoding = self.secondEmbedding(
            positions_vector).to(self.device)

        return self.dropoutLayer(first_embedding + positional_encoding)
