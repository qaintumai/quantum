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

class QuantumNeuralNetworkModel:
    def __init__(self, num_layers, num_wires, inputs, var):
        probabilities = True
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.quantum_nn = qnn_circuit(inputs, var)
        self.model = self._build_model()

    def _build_model(self):
        weights = WeightInitializer.init_weights(self.num_layers, self.num_wires)
        shape_tup = weights.shape
        weight_shapes = {'var': shape_tup}
        qlayer = qml.qnn.TorchLayer(self.quantum_nn, weight_shapes)
        model = [qlayer]
        # model = torch.nn.Sequential(qlayer)
        return model

# Example usage
num_layers = 2
num_wires = 6
qnn_model = QuantumNeuralNetworkModel(num_layers, num_wires, qnn_circuit)
model = qnn_model.model   # This is a quantum model converted into a PyTorch model.
