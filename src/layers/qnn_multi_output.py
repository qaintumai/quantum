# Copyright 2015 The qAIntum.ai Authors. All Rights Reserved.
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

import pennylane as qml
from .quantum_data_encoder import QuantumDataEncoder
from .quantum_layer import QuantumNeuralNetworkLayer

"""
Usage:
To use the qnn_multi_output function, import it as follows:
    from layers.qnn_multi_output import qnn_multi_output

Example:
    inputs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    var = [var1, var2, var3]  # Example list of variables for the quantum layers
    output = qnn_multi_output(inputs, var)
"""
# Define the number of wires and basis states
num_wires = 8
num_basis = 2

# Select a device
dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=num_basis)

@qml.qnode(dev, interface="torch")
def qnn_multi_output(inputs, var):
    """
    This module defines a quantum neural network (QNN) with multiple outputs using PennyLane and PyTorch. The QNN takes input data, encodes it using a quantum data encoder, applies multiple quantum neural network layers, and returns the expectation values of the Pauli-X operator for each wire.

    Parameters:
    - inputs (list or array-like): Input data to be encoded and processed by the QNN.
    - var (list or array-like): List of variables for the quantum layers.

    Returns:
    - list: Expectation values of the Pauli-X operator for each wire.
    """
    # Encode the input data using the quantum data encoder
    encoder = QuantumDataEncoder(num_wires)
    encoder.encode(inputs)

    # Apply iterative quantum layers
    q_layer = QuantumNeuralNetworkLayer(num_wires)
    for v in var:
        q_layer.apply(v)

    # Return the probabilities for all wires
    return [qml.expval(qml.X(wire)) for wire in range(num_wires)]
