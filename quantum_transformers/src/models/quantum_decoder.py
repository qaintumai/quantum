
# Define the QuantumDecoder class
import torch.nn as nn
from src.layers import QuantumFeedForward, MultiHeadedAttention

class QuantumDecoder(nn.Module):
    def __init__(self, embed_len, num_heads, batch_size, dropout=0.1, mask=None):
        super(QuantumDecoder, self).__init__()
        self.embed_len = embed_len
        self.multihead_self_attention = MultiHeadedAttention(
            num_heads, embed_len, batch_size, mask)
        self.multihead_enc_dec_attention = MultiHeadedAttention(
            num_heads, embed_len, batch_size, mask)
        self.first_norm = nn.LayerNorm(self.embed_len)
        self.second_norm = nn.LayerNorm(self.embed_len)
        self.third_norm = nn.LayerNorm(self.embed_len)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.quantum_feed_forward = QuantumFeedForward(embed_len, dropout)

    def forward(self, target, encoder_output):
        # Self attention
        self_attention_output = self.multihead_self_attention(
            target, target, target)
        self_attention_output = self.dropout_layer(self_attention_output)
        first_sublayer_output = self.first_norm(self_attention_output + target)

        # Encoder-decoder attention
        enc_dec_attention_output = self.multihead_enc_dec_attention(
            first_sublayer_output, encoder_output, encoder_output)
        enc_dec_attention_output = self.dropout_layer(enc_dec_attention_output)
        second_sublayer_output = self.second_norm(
            enc_dec_attention_output + first_sublayer_output)

        # Quantum Feed-forward
        return self.quantum_feed_forward(second_sublayer_output)
