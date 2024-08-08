# Copyright 2024 The qAIntum.ai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn

class InputEmbedding(nn.Module):
    """
    A class used to generate embeddings for input data and add positional encodings.

    Usage:
    To use the InputEmbedding class, import it as follows:
    from layers.input_embedding import InputEmbedding

    Example:
    embedding_layer = InputEmbedding(input_vocab_size=10000, embed_len=128)
    output = embedding_layer(input_tensor)
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
