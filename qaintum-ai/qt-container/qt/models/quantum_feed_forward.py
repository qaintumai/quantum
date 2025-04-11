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

    def __init__(self, num_layers, num_wires, quantum_nn, embed_len, dropout=0.1):
        """
        Initializes the QuantumFeedForward class with the given parameters.

        Parameters:
        - embed_len (int): Length of the embedding vector.
        - dropout (float, optional): Dropout rate for regularization. Default is 0.1.
        """
        super(QuantumFeedForward, self).__init__()
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.quantum_nn = quantum_nn
        #TODO: circular imports, refactor
        #from models.quantum_neural_network import QuantumNeuralNetwork
        self.qnn_model = QuantumNeuralNetwork(self.num_layers, self.num_wires, self.quantum_nn).qlayers
        self.quantum_feed_forward = nn.Sequential(self.qnn_model)
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
        ff_output = self.quantum_feed_forward(x)
        ff_output = self.dropout_layer(ff_output)
        return self.layer_norm(ff_output + x)

