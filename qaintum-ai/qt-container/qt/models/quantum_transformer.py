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

# Define the Transformer class
import torch.nn as nn
from qt-container.qt.layers import InputEmbedding
from qt-container.qt.models import QuantumDecoder
from qt-container.qt.models import QuantumEncoder

class QuantumTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_len, num_heads, num_layers, num_wires, quantum_nn, batch_size, vocab_size, dropout=0.1, device='cpu'):
        super(QuantumTransformer, self).__init__()
        self.embed_len = embed_len
        self.device = device
        self.embedding = InputEmbedding(
            vocab_size, embed_len, dropout, device).to(device)
        self.encoder_layers = nn.ModuleList([QuantumEncoder(
            embed_len, num_heads, num_layers, num_wires, quantum_nn, dropout).to(device) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([QuantumDecoder(
            embed_len, num_heads, num_layers, num_wires, quantum_nn, dropout).to(device) for _ in range(num_decoder_layers)])
        self.output_linear = nn.Linear(embed_len, vocab_size).to(device)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        # Encoder forward pass
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(
                encoder_output, encoder_output, encoder_output)

        # Decoder forward pass
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)

        return self.output_linear(decoder_output)
