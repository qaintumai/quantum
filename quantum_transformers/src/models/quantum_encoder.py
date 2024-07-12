# Define the EncoderBlock class
import torch.nn as nn
from layers import QuantumFeedForward, MultiHeadedAttention


class QuantumEncoder(nn.Module):
    def __init__(self, embed_len, num_heads, batch_size, dropout=0.1, mask=None):
        super(QuantumEncoder, self).__init__()
        self.embed_len = embed_len
        self.multihead = MultiHeadedAttention(
            num_heads, embed_len, batch_size, mask)
        self.first_norm = nn.LayerNorm(self.embed_len)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.quantum_feed_forward = QuantumFeedForward(embed_len, dropout)

    def forward(self, queries, keys, values):
        attention_output = self.multihead(queries, keys, values)
        attention_output = self.dropout_layer(attention_output)
        first_sublayer_output = self.first_norm(attention_output + queries)
        return self.quantum_feed_forward(first_sublayer_output)
