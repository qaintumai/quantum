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

# Weight Initializer may not be necessary.

import pennylane as qml
import torch
import sys
import os

# Add the src directory to the Python path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from layers.weight_initializer import WeightInitializer
from layers.qnn_circuit import qnn_circuit

from utils.config import num_wires, num_basis, single_output, multi_output, probabilities

class QuantumNeuralNetwork:
    def __init__(self, num_layers=2, num_modes=6, qnn_circuit=None):
        """
        Initializes the quantum layer model by setting up the weights and converting
        the quantum neural network (qnn) into a Torch layer.

        Parameters:
        - quantum_nn: The quantum neural network function to be converted.
        - num_layers: Number of quantum layers.
        - num_modes: Number of qumodes (wires) for the quantum circuit.
        """
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.qnn_circuit = qnn_circuit

        # Initialize weights for quantum layers
        self.weights = WeightInitializer.init_weights(self.num_layers, self.num_modes)

        # Convert the quantum layer to a Torch layer
        self.qlayers = self._build_quantum_layers()

    def _build_quantum_layers(self):
        """
        Converts the quantum neural network to a Torch layer and returns the layers as a list.
        """
        # Get the shape of the weights and pass them to TorchLayer
        shape_tup = self.weights.shape
        weight_shapes = {'var': shape_tup}

        # Create a TorchLayer from the quantum circuit
        qlayers = qml.qnn.TorchLayer(self.qnn_circuit, weight_shapes)

        # Store the quantum layer in a list (more layers can be added if needed)
        return qlayers
