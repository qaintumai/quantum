import unittest
import torch
from quantum.src.layers.multi_headed_attention import MultiHeadedAttention

class TestMultiHeadedAttention(unittest.TestCase):

    def setUp(self):
        """
        Initialize a MultiHeadedAttention instance and some sample inputs.
        """
        self.num_heads = 8
        self.embed_len = 128
        self.batch_size = 32
        self.seq_len = 10  # Sequence length
        self.multi_head_attention = MultiHeadedAttention(
            num_heads=self.num_heads, embed_len=self.embed_len, batch_size=self.batch_size)

        # Create sample inputs for queries, keys, and values
        self.queries = torch.randn(self.batch_size, self.seq_len, self.embed_len)
        self.keys = torch.randn(self.batch_size, self.seq_len, self.embed_len)
        self.values = torch.randn(self.batch_size, self.seq_len, self.embed_len)

    def test_output_shape(self):
        """
        Test that the output shape of the MultiHeadedAttention layer is as expected.
        """
        output = self.multi_head_attention.forward(self.queries, self.keys, self.values)
        expected_shape = (self.batch_size, self.seq_len, self.embed_len)
        self.assertEqual(output.shape, expected_shape, 
                         f"Expected output shape {expected_shape}, but got {output.shape}")

    def test_attention_with_different_seq_len(self):
        """
        Test that the MultiHeadedAttention layer can handle different sequence lengths.
        """
        # Change sequence length for queries, keys, and values
        new_seq_len = 20
        queries = torch.randn(self.batch_size, new_seq_len, self.embed_len)
        keys = torch.randn(self.batch_size, new_seq_len, self.embed_len)
        values = torch.randn(self.batch_size, new_seq_len, self.embed_len)

        output = self.multi_head_attention.forward(queries, keys, values)
        expected_shape = (self.batch_size, new_seq_len, self.embed_len)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, but got {output.shape}")

    def test_zero_input(self):
        """
        Test that the MultiHeadedAttention layer can handle zero inputs.
        """
        queries = torch.zeros(self.batch_size, self.seq_len, self.embed_len)
        keys = torch.zeros(self.batch_size, self.seq_len, self.embed_len)
        values = torch.zeros(self.batch_size, self.seq_len, self.embed_len)

        output = self.multi_head_attention.forward(queries, keys, values)
        # The output should be close to zero since the input is zero
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)),
                        "Output should be close to zero for zero input.")

    def test_attention_masking(self):
        """
        Test that masking functionality works as expected (if implemented).
        """
        # Initialize with masking enabled
        masked_attention = MultiHeadedAttention(
            num_heads=self.num_heads, embed_len=self.embed_len, batch_size=self.batch_size, mask=True)

        output = masked_attention.forward(self.queries, self.keys, self.values)
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
            self.multi_head_attention.forward(queries, keys, values)


if __name__ == '__main__':
    unittest.main()
