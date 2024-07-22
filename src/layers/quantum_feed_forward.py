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

"""
Author: Sophie Choe, qAIntum.ai
Date: July 17, 2024
Essentially, this is a quantum neural network (QNN). 
This file is a specific QNN as a quantum version of the feed forward block of a transformer.
"""
import torch.nn as nn
# from src.models import 

class QuantumFeedForward(nn.Module):
    def __init__(self, embed_len, dropout=0.1):
        super(QuantumFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_len)

    def forward(self, x):
        ff_output = self.feed_forward(x)
        ff_output = self.dropout_layer(ff_output)
        return self.layer_norm(ff_output + x)


# Example usage
embed_len = 64  # example value
model = QuantumFeedForward(embed_len)

# Calculate the number of parameters


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f'Total number of parameters in FeedForwardBlock: {total_params}')
