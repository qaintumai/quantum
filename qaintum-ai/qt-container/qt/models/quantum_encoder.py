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

# Define the EncoderBlock class
from torch import nn
from qt-container.qt.layers.multi_headed_attention import MultiHeadedAttention
from qt-container.qt.models.quantum_feed_forward import QuantumFeedForward
# from qt-container.qt.layers.qnn_circuit import qnn_circuit

class QuantumEncoder(nn.Module):
    def __init__(self, embed_len, num_heads, num_layers, num_wires, quantum_nn, dropout=0.1, mask=None):
        super(QuantumEncoder, self).__init__()
        self.embed_len = embed_len
        self.multihead = MultiHeadedAttention(num_heads, embed_len, mask)
        self.first_norm = nn.LayerNorm(self.embed_len)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.quantum_feed_forward = QuantumFeedForward(num_layers, num_wires, quantum_nn, embed_len, dropout)

    def forward(self, queries, keys, values):
        attention_output = self.multihead(queries, keys, values)
        attention_output = self.dropout_layer(attention_output)
        first_sublayer_output = self.first_norm(attention_output + queries)
        return self.quantum_feed_forward(first_sublayer_output)

