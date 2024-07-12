# Define the Transformer class
import torch.nn as nn
from layers import InputEmbedding
from models import QuantumEncoder, QuantumDecoder


class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_len, num_heads, batch_size, vocab_size, dropout=0.1, device='cpu'):
        super(Transformer, self).__init__()
        self.embed_len = embed_len
        self.device = device
        self.embedding = InputEmbedding(
            vocab_size, embed_len, dropout, device).to(device)
        self.encoder_layers = nn.ModuleList([QuantumEncoder(
            embed_len, num_heads, batch_size, dropout).to(device) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([QuantumDecoder(
            embed_len, num_heads, batch_size, dropout).to(device) for _ in range(num_decoder_layers)])
        self.output_linear = nn.Linear(embed_len, vocab_size).to(device)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        # Encoder forward pass
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(
                encoder_output, encoder_output, encoder_output)

        # Decoder forward pass
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)

        return self.output_linear(decoder_output)
