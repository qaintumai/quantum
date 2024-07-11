import math
import torch
import torch.nn as nn

# Define the ScaledDotProduct class


class ScaledDotProduct(nn.Module):
    def __init__(self, embed_len, mask=None):
        super(ScaledDotProduct, self).__init__()
        self.embed_len = embed_len
        self.mask = mask
        self.dk = embed_len  # dimension of keys and queries
        # Apply softmax on the last dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        compatibility = torch.matmul(queries, keys.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.dk)
        compatibility = self.softmax(compatibility)

        if self.mask is not None:
            compatibility = torch.tril(compatibility)

        return torch.matmul(compatibility, values)

