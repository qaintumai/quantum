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

import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from layers.multi_headed_attention import MultiHeadedAttention

class TestMultiHeadedAttention(unittest.TestCase):

    def setUp(self):
        """
        Initialize a MultiHeadedAttention instance and some sample inputs.
        """
        self.num_heads = 8
        self.embed_len = 64
        self.seq_len = 10
        self.batch_size = 32
        self.mask = None

        # Create an instance of MultiHeadedAttention
        self.multi_head_attention = MultiHeadedAttention(
            num_heads=self.num_heads, embed_len=self.embed_len, mask=self.mask
        )

        # Create sample inputs for queries, keys, and values
        self.queries = torch.rand(self.batch_size, self.seq_len, self.embed_len)
        self.keys = torch.rand(self.batch_size, self.seq_len, self.embed_len)
        self.values = torch.rand(self.batch_size, self.seq_len, self.embed_len)

    def test_output_shape(self):
        """
        Test that the output shape of the MultiHeadedAttention layer is as expected.
        """
        output = self.multi_head_attention(self.queries, self.keys, self.values)
        expected_shape = (self.batch_size, self.seq_len, self.embed_len)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, but got {output.shape}")

    def test_attention_with_different_seq_len(self):
        """
        Test that the MultiHeadedAttention layer can handle different sequence lengths.
        """
        # Change sequence length for queries, keys, and values
        new_seq_len = 20
        queries = torch.rand(self.batch_size, new_seq_len, self.embed_len)
        keys = torch.rand(self.batch_size, new_seq_len, self.embed_len)
        values = torch.rand(self.batch_size, new_seq_len, self.embed_len)

        output = self.multi_head_attention(queries, keys, values)
        expected_shape = (self.batch_size, new_seq_len, self.embed_len)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, but got {output.shape}")

    def test_attention_masking(self):
        """
        Test that masking functionality works as expected (if implemented).
        """
        # Initialize with masking enabled (assuming masking can be handled in your implementation)
        masked_attention = MultiHeadedAttention(
            num_heads=self.num_heads, embed_len=self.embed_len, mask=True
        )

        output = masked_attention(self.queries, self.keys, self.values)
        expected_shape = (self.batch_size, self.seq_len, self.embed_len)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, but got {output.shape}")

    def test_forward_invalid_input(self):
        """
        Test that the MultiHeadedAttention layer raises an error when given invalid inputs.
        """
        # Provide mismatched dimensions for queries, keys, values
        queries = torch.randn(self.batch_size, self.seq_len + 2, self.embed_len)
        keys = torch.randn(self.batch_size, self.seq_len, self.embed_len)
        values = torch.randn(self.batch_size, self.seq_len, self.embed_len)

        with self.assertRaises(RuntimeError):
            self.multi_head_attention(queries, keys, values)


if __name__ == '__main__':
    unittest.main()
