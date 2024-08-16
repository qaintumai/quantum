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

"""
Quantum Transformer Example

This script demonstrates a Quantum Transformer.
"""

import torch
import torch.nn as nn
import math

import numpy as np
import pennylane as qml

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Define the InputEmbedding class
class InputEmbedding(nn.Module):
    def __init__(self, input_vocab_size, embed_len, dropout=0.1, device='cpu'):
        super(InputEmbedding, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_len = embed_len
        self.dropout = dropout
        self.device = device

        self.firstEmbedding = nn.Embedding(self.input_vocab_size, self.embed_len).to(self.device)
        self.secondEmbedding = nn.Embedding(self.input_vocab_size, self.embed_len).to(self.device)
        self.dropoutLayer = nn.Dropout(p=self.dropout)

    def forward(self, input):
        first_embedding = self.firstEmbedding(input).to(self.device)
        batch_size, seq_len = input.shape

        positions_vector = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        positional_encoding = self.secondEmbedding(positions_vector).to(self.device)

        return self.dropoutLayer(first_embedding + positional_encoding)

def test_input_embedding():
    # Define parameters
    input_vocab_size = 100
    embed_len = 64
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    device = 'cpu'

    # Create an instance of InputEmbedding
    model = InputEmbedding(input_vocab_size, embed_len, dropout, device)

    # Create a dummy input tensor with random integers in the range of the vocabulary size
    dummy_input = torch.randint(0, input_vocab_size, (batch_size, seq_len)).to(device)

    # Forward pass
    output = model(dummy_input)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape

# Run the test
test_input_embedding()

# Define the ScaledDotProduct class
class ScaledDotProduct(nn.Module):
    def __init__(self, embed_len, mask=None):
        super(ScaledDotProduct, self).__init__()
        self.embed_len = embed_len
        self.mask = mask
        self.dk = embed_len  # dimension of keys and queries
        self.softmax = nn.Softmax(dim=-1)  # Apply softmax on the last dimension

    def forward(self, queries, keys, values):
        compatibility = torch.matmul(queries, keys.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.dk)
        compatibility = self.softmax(compatibility)

        if self.mask is not None:
            compatibility = torch.tril(compatibility)

        return torch.matmul(compatibility, values)

# Test the ScaledDotProduct class
def test_scaled_dot_product():
    # Define parameters
    embed_len = 64
    seq_len = 10
    batch_size = 32

    # Create an instance of ScaledDotProduct
    model = ScaledDotProduct(embed_len)

    # Create dummy input tensors
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(queries, keys, values)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")
    return output.shape

# Run the test
test_scaled_dot_product()

# Define the MultiHeadedAttention class
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embed_len, batch_size, mask=None):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_len = embed_len
        self.batch_size = batch_size
        self.mask = mask
        self.head_length = int(self.embed_len / self.num_heads)
        self.q_in = self.v_in = self.k_in = self.embed_len

        self.q_linear = nn.Linear(int(self.q_in), int(self.q_in))
        self.k_linear = nn.Linear(int(self.k_in), int(self.k_in))
        self.v_linear = nn.Linear(int(self.v_in), int(self.v_in))

        if self.mask is not None:
            self.attention = ScaledDotProduct(embed_len=self.head_length, mask=True)
        else:
            self.attention = ScaledDotProduct(embed_len=self.head_length)

        self.output_linear = nn.Linear(self.q_in, self.q_in)

    def forward(self, queries, keys, values):
        queries = self.q_linear(queries).reshape(self.batch_size, -1, self.num_heads, self.head_length)
        queries = queries.transpose(1, 2)

        keys = self.k_linear(keys).reshape(self.batch_size, -1, self.num_heads, self.head_length)
        keys = keys.transpose(1, 2)

        values = self.v_linear(values).reshape(self.batch_size, -1, self.num_heads, self.head_length)
        values = values.transpose(1, 2)

        sdp_output = self.attention(queries, keys, values).transpose(1, 2).reshape(self.batch_size, -1, self.num_heads * self.head_length)

        return self.output_linear(sdp_output)

# Test the MultiHeadedAttention class
def test_multi_headed_attention():
    # Define parameters
    num_heads = 8
    embed_len = 64
    seq_len = 10
    batch_size = 32
    mask = None

    # Create an instance of MultiHeadedAttention
    model = MultiHeadedAttention(num_heads, embed_len, batch_size, mask)

    # Create dummy input tensors
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(queries, keys, values)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape

# Run the test
test_multi_headed_attention()

# Define the DataEncoding class
class DataEncoding:
    def __init__(self, num_wires):
        self.num_wires = num_wires

    def encode(self, x):
        num_features = len(x)

        # Squeezing gates
        for i in range(0, min(num_features, self.num_wires * 2), 2):
            qml.Squeezing(x[i], x[i + 1], wires=i // 2)

        # Beamsplitter gates
        for i in range(self.num_wires - 1):
            idx = self.num_wires * 2 + i * 2
            if idx + 1 < num_features:
                qml.Beamsplitter(x[idx], x[idx + 1], wires=[i % self.num_wires, (i + 1) % self.num_wires])

        # Rotation gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + i
            if idx < num_features:
                qml.Rotation(x[idx], wires=i)

        # Displacement gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + i * 2
            if idx + 1 < num_features:
                qml.Displacement(x[idx], x[idx + 1], wires=i)

        # Kerr gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + i
            if idx < num_features:
                qml.Kerr(x[idx], wires=i)

        # Squeezing gates (second set)
        for i in range(0, min(num_features - (self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires), self.num_wires * 2), 2):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires + i
            if idx + 1 < num_features:
                qml.Squeezing(x[idx], x[idx + 1], wires=i // 2)

        # Rotation gates (second set)
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires + self.num_wires * 2 + i
            if idx < num_features:
                qml.Rotation(x[idx], wires=i)

# Define the QuantumLayer class
class QuantumLayer:
    def __init__(self, num_wires):
        self.num_wires = num_wires

    def apply_layer(self, v):
        num_params = len(v)

        # Interferometer 1
        for i in range(self.num_wires - 1):
            idx = i * 2
            if idx + 1 < num_params:
                theta = v[idx]
                phi = v[idx + 1]
                qml.Beamsplitter(theta, phi, wires=[i % self.num_wires, (i + 1) % self.num_wires])

        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + i
            if idx < num_params:
                qml.Rotation(v[idx], wires=i)

        # Squeezers
        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + i
            if idx < num_params:
                qml.Squeezing(v[idx], 0.0, wires=i)

        # Interferometer 2
        for i in range(self.num_wires - 1):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + i * 2
            if idx + 1 < num_params:
                theta = v[idx]
                phi = v[idx + 1]
                qml.Beamsplitter(theta, phi, wires=[i % self.num_wires, (i + 1) % self.num_wires])

        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + (self.num_wires - 1) * 2 + i
            if idx < num_params:
                qml.Rotation(v[idx], wires=i)

        # Bias addition
        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + (self.num_wires - 1) * 2 + self.num_wires + i
            if idx < num_params:
                qml.Displacement(v[idx], 0.0, wires=i)

        # Non-linear activation function
        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + i
            if idx < num_params:
                qml.Kerr(v[idx], wires=i)

# Define the WeightInitializer class
class WeightInitializer:
    @staticmethod
    def init_weights(layers, modes, active_sd=0.0001, passive_sd=0.1):
        M = (modes - 1) * 2 + modes  # Number of interferometer parameters

        int1_weights = np.random.normal(size=[layers, M], scale=passive_sd)
        s_weights = np.random.normal(size=[layers, modes], scale=active_sd)
        int2_weights = np.random.normal(size=[layers, M], scale=passive_sd)
        dr_weights = np.random.normal(size=[layers, modes], scale=active_sd)
        k_weights = np.random.normal(size=[layers, modes], scale=active_sd)

        weights = np.concatenate([int1_weights, s_weights, int2_weights, dr_weights, k_weights], axis=1)

        return weights

# Think through the output
num_modes = 6
num_basis = 2

# Select a device
dev = qml.device("strawberryfields.fock", wires=num_modes, cutoff_dim=num_basis)

@qml.qnode(dev, interface="torch")
def quantum_nn(inputs, var):
    num_wires = 6
    encoder = DataEncoding(num_wires)
    encoder.encode(inputs)

    # Iterative quantum layers
    q_layer = QuantumLayer(num_wires)
    for v in var:
        q_layer.apply_layer(v)

    # Return the probabilities
    return qml.probs(wires=[0, 1, 2, 3, 4, 5])

num_layers = 2

# Initialize weights for quantum layers
weights = WeightInitializer.init_weights(num_layers, num_modes)

# Convert the quantum layer to a Torch layer
shape_tup = weights.shape
weight_shapes = {'var': shape_tup}

qlayer = qml.qnn.TorchLayer(quantum_nn, weight_shapes)
layers = [qlayer]

# Define the FeedForwardBlock class
class FeedForwardBlock(nn.Module):
    def __init__(self, embed_len, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.feed_forward = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_len)

    def forward(self, x):
        ff_output = self.feed_forward(x)
        ff_output = self.dropout_layer(ff_output)
        return self.layer_norm(ff_output + x)

# Example usage
embed_len = 64  # example value
model = FeedForwardBlock(embed_len)

# Calculate the number of parameters
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f'Total number of parameters in FeedForwardBlock: {total_params}')

# Test the FeedForwardBlock class
def test_feed_forward_block():
    # Define parameters
    embed_len = 64
    seq_len = 10
    batch_size = 32
    dropout = 0.1

    # Create an instance of FeedForwardBlock
    model = FeedForwardBlock(embed_len, dropout)

    # Create a dummy input tensor
    dummy_input = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(dummy_input)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape

# Run the test
test_feed_forward_block()

# Define the EncoderBlock class
class EncoderBlock(nn.Module):
    def __init__(self, embed_len, num_heads, batch_size, dropout=0.1, mask=None):
        super(EncoderBlock, self).__init__()
        self.embed_len = embed_len
        self.multihead = MultiHeadedAttention(num_heads, embed_len, batch_size, mask)
        self.first_norm = nn.LayerNorm(self.embed_len)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.feed_forward_block = FeedForwardBlock(embed_len, dropout)

    def forward(self, queries, keys, values):
        attention_output = self.multihead(queries, keys, values)
        attention_output = self.dropout_layer(attention_output)
        first_sublayer_output = self.first_norm(attention_output + queries)
        return self.feed_forward_block(first_sublayer_output)

# Test the EncoderBlock class
def test_encoder_block():
    # Define parameters
    embed_len = 64
    num_heads = 8
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    mask = None

    # Create an instance of EncoderBlock
    model = EncoderBlock(embed_len, num_heads, batch_size, dropout, mask)

    # Create dummy input tensors
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(queries, keys, values)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape

# Run the test
test_encoder_block()

# Define the DecoderBlock class
class DecoderBlock(nn.Module):
    def __init__(self, embed_len, num_heads, batch_size, dropout=0.1, mask=None):
        super(DecoderBlock, self).__init__()
        self.embed_len = embed_len
        self.multihead_self_attention = MultiHeadedAttention(num_heads, embed_len, batch_size, mask)
        self.multihead_enc_dec_attention = MultiHeadedAttention(num_heads, embed_len, batch_size, mask)
        self.first_norm = nn.LayerNorm(self.embed_len)
        self.second_norm = nn.LayerNorm(self.embed_len)
        self.third_norm = nn.LayerNorm(self.embed_len)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.feed_forward_block = FeedForwardBlock(embed_len, dropout)

    def forward(self, target, encoder_output):
        # Self attention
        self_attention_output = self.multihead_self_attention(target, target, target)
        self_attention_output = self.dropout_layer(self_attention_output)
        first_sublayer_output = self.first_norm(self_attention_output + target)

        # Encoder-decoder attention
        enc_dec_attention_output = self.multihead_enc_dec_attention(first_sublayer_output, encoder_output, encoder_output)
        enc_dec_attention_output = self.dropout_layer(enc_dec_attention_output)
        second_sublayer_output = self.second_norm(enc_dec_attention_output + first_sublayer_output)

        # Feed-forward
        return self.feed_forward_block(second_sublayer_output)

# Test the DecoderBlock class
def test_decoder_block():
    # Define parameters
    embed_len = 64
    num_heads = 8
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    mask = None

    # Create an instance of DecoderBlock
    model = DecoderBlock(embed_len, num_heads, batch_size, dropout, mask)

    # Create dummy input tensors
    target = torch.rand(batch_size, seq_len, embed_len)
    encoder_output = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(target, encoder_output)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape

# Run the test
test_decoder_block()

# Define the Transformer class
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_len, num_heads, batch_size, vocab_size, dropout=0.1, device='cpu'):
        super(Transformer, self).__init__()
        self.embed_len = embed_len
        self.device = device
        self.embedding = InputEmbedding(vocab_size, embed_len, dropout, device).to(device)
        self.encoder_layers = nn.ModuleList([EncoderBlock(embed_len, num_heads, batch_size, dropout).to(device) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(embed_len, num_heads, batch_size, dropout).to(device) for _ in range(num_decoder_layers)])
        self.output_linear = nn.Linear(embed_len, vocab_size).to(device)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        # Encoder forward pass
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, encoder_output, encoder_output)

        # Decoder forward pass
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)

        return self.output_linear(decoder_output)

# Test the Transformer class
def test_transformer():
    # Define parameters
    num_encoder_layers = 6
    num_decoder_layers = 6
    embed_len = 64
    num_heads = 8
    seq_len = 20
    batch_size = 32
    vocab_size = 100
    dropout = 0.1
    device = 'cpu'

    # Create an instance of Transformer
    model = Transformer(num_encoder_layers, num_decoder_layers, embed_len, num_heads, batch_size, vocab_size, dropout, device)

    # Create dummy input tensors
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    output = model(src, tgt)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, vocab_size), f"Expected output shape {(batch_size, seq_len, vocab_size)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")
    return output.shape

# Run the test
test_transformer()

