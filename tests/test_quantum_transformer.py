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

# Test the Transformer class
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.quantum_transformer import Transformer


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
    model = Transformer(num_encoder_layers, num_decoder_layers,
                        embed_len, num_heads, batch_size, vocab_size, dropout, device)

    # Create dummy input tensors
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    output = model(src, tgt)

    # Check the output shape
    assert output.shape == (
        batch_size, seq_len, vocab_size), f"Expected output shape {(batch_size, seq_len, vocab_size)}, but got {output.shape}"

    # Check the output type
    assert isinstance(
        output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")


def main():
    # Run all tests
    test_transformer()


if __name__ == '__main__':
    main()