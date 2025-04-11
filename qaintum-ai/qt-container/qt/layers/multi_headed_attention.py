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

from torch import nn
from qt-container.qt.layers.scaled_dot_product import ScaledDotProduct
import torch

class MultiHeadedAttention(nn.Module):
    """
    A class used to implement the multi-headed attention mechanism,
    splitting the input into multiple heads, applying scaled dot-product
    attention to each head, and then concatenating the results.

    Usage:
    To use the MultiHeadedAttention class, import it as follows:
    from layers.multi_headed_attention import MultiHeadedAttention

    Example:
    attention_layer = MultiHeadedAttention(num_heads=8, embed_len=128)
    output = attention_layer(queries, keys, values)
    """

    def __init__(self, num_heads, embed_len, mask=None):
        """
        Initializes the MultiHeadedAttention class with the given parameters.

        Parameters:
        - num_heads (int): Number of attention heads.
        - embed_len (int): Length of the embedding vector.
        - mask (bool, optional): Whether to apply masking. Default is None.
        """
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_len = embed_len
        self.mask = mask
        self.head_length = int(self.embed_len / self.num_heads)
        self.q_in = self.v_in = self.k_in = self.embed_len

        # Define linear layers for queries, keys, and values (with bias enabled by default)
        self.q_linear = nn.Linear(self.q_in, self.q_in)
        self.k_linear = nn.Linear(self.k_in, self.k_in)
        self.v_linear = nn.Linear(self.v_in, self.v_in)

        # Define the scaled dot-product attention mechanism with optional masking
        self.attention = ScaledDotProduct(embed_len=self.head_length, mask=self.mask is not None)

        # Define the output linear layer (with bias enabled by default)
        self.output_linear = nn.Linear(self.q_in, self.q_in)

    def forward(self, queries, keys, values):
        """
        Computes the multi-headed attention output.

        Parameters:
        - queries (torch.Tensor): Tensor containing the queries (batch_size, seq_len, embed_len).
        - keys (torch.Tensor): Tensor containing the keys (batch_size, seq_len, embed_len).
        - values (torch.Tensor): Tensor containing the values (batch_size, seq_len, embed_len).

        Returns:
        - torch.Tensor: Tensor containing the multi-headed attention output.
        """
        # Dynamically infer the batch size from the input tensors
        batch_size = queries.size(0)

        # Dimension check to ensure compatibility between queries, keys, and values
        if queries.size(1) != keys.size(1) or queries.size(1) != values.size(1):
            raise RuntimeError("Mismatched dimensions between queries, keys, and values.")

        # Linear transformation and reshaping of queries, keys, and values
        queries = self.q_linear(queries).reshape(batch_size, -1, self.num_heads, self.head_length)
        queries = queries.transpose(1, 2)

        keys = self.k_linear(keys).reshape(batch_size, -1, self.num_heads, self.head_length)
        keys = keys.transpose(1, 2)

        values = self.v_linear(values).reshape(batch_size, -1, self.num_heads, self.head_length)
        values = values.transpose(1, 2)

        # Apply scaled dot-product attention and reshape the output
        sdp_output = self.attention(queries, keys, values).transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_length)

        return self.output_linear(sdp_output)
