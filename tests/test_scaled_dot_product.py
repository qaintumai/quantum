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
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from layers.scaled_dot_product import ScaledDotProduct


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
    start_time = time.time()
    output = model(queries, keys, values)
    elapsed_time = time.time() - start_time

    # Check the output shape
    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected output type torch.Tensor, but got {type(output)}"

    # Check performance: Assert the forward pass is reasonably fast
    assert elapsed_time < 1.0, f"Forward pass took too long: {elapsed_time:.4f} seconds"

    print("Test passed!")

    return output.shape


def test_edge_cases():
    # Edge case: Small tensors
    embed_len = 1
    seq_len = 1
    batch_size = 1

    model = ScaledDotProduct(embed_len)

    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    output = model(queries, keys, values)

    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"
    print("Edge case for small tensors passed!")

    # Edge case: Large tensors
    embed_len = 512
    seq_len = 1000
    batch_size = 64

    model = ScaledDotProduct(embed_len)

    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    output = model(queries, keys, values)

    assert output.shape == (
        batch_size, seq_len, embed_len), f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"
    print("Edge case for large tensors passed!")


if __name__ == '__main__':
    shape = test_scaled_dot_product()
    test_edge_cases()

