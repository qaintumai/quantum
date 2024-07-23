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
from layers import WeightInitializer, qnn

class QuantumNeuralNetworkModel:
    def __init__(self, num_layers, num_wires, quantum_nn):
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.quantum_nn = quantum_nn
        self.model = self._build_model()

    def _build_model(self):
        weights = WeightInitializer.init_weights(self.num_layers, self.num_wires)
        shape_tup = weights.shape
        weight_shapes = {'var': shape_tup}
        qlayer = qml.qnn.TorchLayer(self.quantum_nn, weight_shapes)
        model = [qlayer]
        return model

# Example usage
num_layers = 2
num_wires = 6
qnn_model = QuantumNeuralNetworkModel(num_layers, num_wires, qnn)
model = qnn_model.model
