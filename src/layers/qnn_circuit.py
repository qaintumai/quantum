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

import pennylane as qml
from .quantum_data_encoder import QuantumDataEncoder
from .quantum_layer import QuantumNeuralNetworkLayer

#model output type
single_output = True
multi_output = False
probabilities = False


# Define the number of wires and basis states
num_basis = 2
num_wires = 8
if (single_output):
    num_wires = 6

# Select a device
dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=num_basis)

@qml.qnode(dev, interface="torch")
def qnn_circuit(inputs, var):
    """
    This module defines a quantum neural network (QNN) that can return multiple outputs,
    a single output, or a probability distribution using PennyLane and PyTorch. The QNN
    takes input data, encodes it using a quantum data encoder, applies multiple quantum
    neural network layers, and returns the specified output based on the structure of 'var'.

    Parameters:
    - inputs (list or array-like): Input data to be encoded and processed by the QNN.
    - var (list or array-like): List of variables for the quantum layers, structure determines
    output type.

    Returns:
    - list or float: The specified output type.
    """
    encoder = QuantumDataEncoder(num_wires)
    encoder.encode(inputs)

    # Iterative quantum layers
    q_layer = QuantumNeuralNetworkLayer(num_wires)
    for v in var:
        q_layer.apply(v)

    if multi_output:
        # Return the probabilities for all wires
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    if probabilities:
        # Return the probabilities NOTE: not functional, need to review pennylane function.
        wires = [0,1,2,3,4,5,6,7]
        return [qml.probs(wires=wires)]

    #else model output type is single
    return qml.expval(qml.X(0))
