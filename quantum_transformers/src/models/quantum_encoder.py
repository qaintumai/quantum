# Define the EncoderBlock class
import torch.nn as nn
from models import FeedForwardBlock, MultiHeadedAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_len, num_heads, batch_size, dropout=0.1, mask=None):
        super(EncoderBlock, self).__init__()
        self.embed_len = embed_len
        self.multihead = MultiHeadedAttention(
            num_heads, embed_len, batch_size, mask)
        self.first_norm = nn.LayerNorm(self.embed_len)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.feed_forward_block = FeedForwardBlock(embed_len, dropout)

    def forward(self, queries, keys, values):
        attention_output = self.multihead(queries, keys, values)
        attention_output = self.dropout_layer(attention_output)
        first_sublayer_output = self.first_norm(attention_output + queries)
        return self.feed_forward_block(first_sublayer_output)
