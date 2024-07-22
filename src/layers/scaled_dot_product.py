# Copyright 2015 The qAIntum.ai Authors. All Rights Reserved.
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
import torch.nn as nn

# Define the ScaledDotProduct class


class ScaledDotProduct(nn.Module):
    def __init__(self, embed_len, mask=None):
        super(ScaledDotProduct, self).__init__()
        self.embed_len = embed_len
        self.mask = mask
        self.dk = embed_len  # dimension of keys and queries
        # Apply softmax on the last dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        compatibility = torch.matmul(queries, keys.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.dk)
        compatibility = self.softmax(compatibility)

        if self.mask is not None:
            compatibility = torch.tril(compatibility)

        return torch.matmul(compatibility, values)
