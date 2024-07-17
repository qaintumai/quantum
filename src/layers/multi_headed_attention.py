import torch.nn as nn
from .scaled_dot_product import ScaledDotProduct

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
            self.attention = ScaledDotProduct(
                embed_len=self.head_length, mask=True)
        else:
            self.attention = ScaledDotProduct(embed_len=self.head_length)

        self.output_linear = nn.Linear(self.q_in, self.q_in)

    def forward(self, queries, keys, values):
        queries = self.q_linear(queries).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        queries = queries.transpose(1, 2)

        keys = self.k_linear(keys).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        keys = keys.transpose(1, 2)

        values = self.v_linear(values).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        values = values.transpose(1, 2)

        sdp_output = self.attention(queries, keys, values).transpose(
            1, 2).reshape(self.batch_size, -1, self.num_heads * self.head_length)

        return self.output_linear(sdp_output)
