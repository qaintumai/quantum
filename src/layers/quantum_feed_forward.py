"""
Author: Sophie Choe, qAIntum.ai
Date: July 17, 2024
Essentially, this is a quantum neural network (QNN). 
This file is a specific QNN as a quantum version of the feed forward block of a transformer.
"""

class QuantumFeedForward(nn.Module):
    def __init__(self, embed_len, dropout=0.1):
        super(QuantumFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_len)

    def forward(self, x):
        ff_output = self.feed_forward(x)
        ff_output = self.dropout_layer(ff_output)
        return self.layer_norm(ff_output + x)


# Example usage
embed_len = 64  # example value
model = QuantumFeedForward(embed_len)

# Calculate the number of parameters


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f'Total number of parameters in FeedForwardBlock: {total_params}')
