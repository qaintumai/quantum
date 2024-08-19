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

# Test the EncoderBlock class
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models import QuantumEncoder


def test_encoder_block():
    # Define parameters
    embed_len = 64
    num_heads = 8
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    mask = None

    # Create an instance of EncoderBlock
    model = QuantumEncoder(embed_len, num_heads, batch_size, dropout, mask)

    # Create dummy input tensors
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(queries, keys, values)

    # Check the output shape
    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(
        output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")


def main():
    # Run all tests
    test_encoder_block()


if __name__ == '__main__':
    main()

