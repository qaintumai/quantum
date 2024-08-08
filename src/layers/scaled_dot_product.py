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

import math
import torch
from torch import nn

class ScaledDotProduct(nn.Module):
    """
    A class used to compute the scaled dot-product attention.

    Usage:
    To use the ScaledDotProduct class, import it as follows:
    from layers.scaled_dot_product import ScaledDotProduct

    Example:
    scaled_dot_product = ScaledDotProduct(embed_len=128, mask=None)
    output = scaled_dot_product(queries, keys, values)
    """

    def __init__(self, embed_len, mask=None):
        """
        Initializes the ScaledDotProduct class with the given parameters.

        Parameters:
        - embed_len (int): Length of the embedding vector (dimension of keys and queries).
        - mask (torch.Tensor, optional): Masking tensor for the attention mechanism. Default is None.
        """
        super(ScaledDotProduct, self).__init__()
        self.embed_len = embed_len
        self.mask = mask
        self.dk = embed_len  # Dimension of keys and queries
        self.softmax = nn.Softmax(dim=-1)  # Apply softmax on the last dimension

    def forward(self, queries, keys, values):
        """
        Computes the scaled dot-product attention.

        Parameters:
        - queries (torch.Tensor): Tensor containing the query vectors.
        - keys (torch.Tensor): Tensor containing the key vectors.
        - values (torch.Tensor): Tensor containing the value vectors.

        Returns:
        - torch.Tensor: Tensor containing the output of the scaled dot-product attention.
        """
        compatibility = torch.matmul(queries, keys.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.dk)
        compatibility = self.softmax(compatibility)

        if self.mask is not None:
            compatibility = torch.tril(compatibility)

        return torch.matmul(compatibility, values)
