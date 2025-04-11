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
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from layers.input_embedding import InputEmbedding
from utils.config import get_device


def test_input_embedding():
    # Define parameters
    input_vocab_size = 100
    embed_len = 64
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    # device = 'cpu'
    device = get_device()

    # Create an instance of InputEmbedding
    model = InputEmbedding(input_vocab_size, embed_len, dropout, device)

    # Create a dummy input tensor with random integers in the range of the vocabulary size
    dummy_input = torch.randint(
        0, input_vocab_size, (batch_size, seq_len)).to(device)

    # Forward pass
    output = model(dummy_input)

    # Check the output shape
    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(
        output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    print("Test passed!")

    return output.shape


if __name__ == '__main__':
    test_input_embedding()
