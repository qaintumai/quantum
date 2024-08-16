# Copyright 2024 The qAIntum.ai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from torch import nn
from models.quantum_neural_network import QuantumNeuralNetwork

class QuantumFeedForward(nn.Module):
    """
    A class used to define a feedforward block for a quantum neural network.

    Usage:
    To use the QuantumFeedForward class, import it as follows:
    from layers.quantum_feed_forward import QuantumFeedForward

    Example:
    model = QuantumFeedForward(embed_len=64)
    output = model(input_tensor)
    """

    def __init__(self, embed_len, dropout=0.1):
        """
        Initializes the QuantumFeedForward class with the given parameters.

        Parameters:
        - embed_len (int): Length of the embedding vector.
        - dropout (float, optional): Dropout rate for regularization. Default is 0.1.
        """
        super(QuantumFeedForward, self).__init__()

        layers = QuantumNeuralNetworkModel(num_layers, num_wires, qnn_circuit).model
        self.feed_forward = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_len)

    def forward(self, x):
        """
        Applies the feedforward block to the input tensor.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying feedforward, dropout, and layer normalization.
        """
        ff_output = self.feed_forward(x)
        ff_output = self.dropout_layer(ff_output)
        return self.layer_norm(ff_output + x)


# Example usage
embed_len = 64  # example value
model = QuantumFeedForward(embed_len)

# Calculate the number of parameters
def count_parameters(module):
    """
    Counts the number of trainable parameters in a module.

    Parameters:
    - module (torch.nn.Module): The module to count parameters for.

    Returns:
    - int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f'Total number of parameters in FeedForwardBlock: {total_params}')
