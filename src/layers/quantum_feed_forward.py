"""
Author: Sophie Choe, qAIntum.ai
Date: July 17, 2024
Essentially, this is a quantum neural network (QNN). 
This file is a specific QNN as a quantum version of the feed forward block of a transformer.
"""

import torch.nn as nn
import pennylane as qml

from models import DataEncoding
from models import QuantumLayer
from models import WeightInitializer

# Think through the output
num_modes = 6
num_basis = 2

# Select a device
dev = qml.device("strawberryfields.fock",
                 wires=num_modes, cutoff_dim=num_basis)


@qml.qnode(dev, interface="torch")
def quantum_nn(inputs, var):
    num_wires = 6
    encoder = DataEncoding(num_wires)
    encoder.encode(inputs)

    # Iterative quantum layers
    q_layer = QuantumLayer(num_wires)
    for v in var:
        q_layer.apply_layer(v)

    # Return the probabilities
    return qml.probs(wires=[0, 1, 2, 3, 4, 5])


num_layers = 2

# Initialize weights for quantum layers
weights = WeightInitializer.init_weights(num_layers, num_modes)

# Convert the quantum layer to a Torch layer
shape_tup = weights.shape
weight_shapes = {'var': shape_tup}

qlayer = qml.qnn.TorchLayer(quantum_nn, weight_shapes)
layers = [qlayer]

# Define the FeedForwardBlock class


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_len, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.feed_forward = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_len)

    def forward(self, x):
        ff_output = self.feed_forward(x)
        ff_output = self.dropout_layer(ff_output)
        return self.layer_norm(ff_output + x)


# Example usage
embed_len = 64  # example value
model = FeedForwardBlock(embed_len)

# Calculate the number of parameters


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f'Total number of parameters in FeedForwardBlock: {total_params}')
