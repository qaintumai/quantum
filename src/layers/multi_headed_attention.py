import torch.nn as nn
from .scaled_dot_product import ScaledDotProduct

"""
Usage:
To use the MultiHeadedAttention class, import it as follows:
    from layers.multi_headed_attention import MultiHeadedAttention

Example:
    attention_layer = MultiHeadedAttention(num_heads=8, embed_len=128, batch_size=32)
    output = attention_layer(queries, keys, values
"""
class MultiHeadedAttention(nn.Module):
    """
    A class used to implement the multi-headed attention mechanism, splitting the input into multiple heads, applies scaled dot-product attention to each head, and then concatenates the results.
    """
    
    def __init__(self, num_heads, embed_len, batch_size, mask=None):
        """
        Initializes the MultiHeadedAttention class with the given parameters.

        Parameters:
        - num_heads (int): Number of attention heads.
        - embed_len (int): Length of the embedding vector.
        - batch_size (int): Size of the batch.
        - mask (bool, optional): Whether to apply masking. Default is None.
        """

        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_len = embed_len
        self.batch_size = batch_size
        self.mask = mask
        self.head_length = int(self.embed_len / self.num_heads)
        self.q_in = self.v_in = self.k_in = self.embed_len

        # Define linear layers for queries, keys, and values
        self.q_linear = nn.Linear(int(self.q_in), int(self.q_in))
        self.k_linear = nn.Linear(int(self.k_in), int(self.k_in))
        self.v_linear = nn.Linear(int(self.v_in), int(self.v_in))

        # Define the scaled dot-product attention mechanism with optional masking
        if self.mask is not None:
            self.attention = ScaledDotProduct(
                embed_len=self.head_length, mask=True)
        else:
            self.attention = ScaledDotProduct(embed_len=self.head_length)
        
        # Define the output linear layer
        self.output_linear = nn.Linear(self.q_in, self.q_in)

    def forward(self, queries, keys, values):
        """
        Computes the multi-headed attention output.

        Parameters:
        - queries (torch.Tensor): Tensor containing the queries.
        - keys (torch.Tensor): Tensor containing the keys.
        - values (torch.Tensor): Tensor containing the values.

        Returns:
        - torch.Tensor: Tensor containing the multi-headed attention output.
        """
        # Linear transformation and reshaping of queries, keys, and values
        queries = self.q_linear(queries).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        queries = queries.transpose(1, 2)

        keys = self.k_linear(keys).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        keys = keys.transpose(1, 2)

        values = self.v_linear(values).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        values = values.transpose(1, 2)

        # Apply scaled dot-product attention and reshape the output
        sdp_output = self.attention(queries, keys, values).transpose(
            1, 2).reshape(self.batch_size, -1, self.num_heads * self.head_length)

        return self.output_linear(sdp_output)
