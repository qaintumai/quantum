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

# Define the number of wires and basis states
num_wires = 6
num_basis = 2

# Select a device
dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=num_basis)

@qml.qnode(dev, interface="torch")
def qnn_single_output(inputs, var):
    encoder = QuantumDataEncoder(num_wires)
    encoder.encode(inputs)

    # Iterative quantum layers
    q_layer = QuantumNeuralNetworkLayer(num_wires)
    for v in var:
        q_layer.apply(v)

    # Return the probabilities
    return qml.expval(qml.X(0))
