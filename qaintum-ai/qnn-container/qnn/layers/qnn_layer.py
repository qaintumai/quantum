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

class QuantumNeuralNetworkLayer:
    """Defines a Quantum Neural Network Layer for photonic quantum computing."""

    def __init__(self, num_wires):
        """
        Initializes the QuantumNeuralNetworkLayer.

        Parameters:
        - num_wires (int): Number of wires (qumodes) in the quantum circuit.
        """
        self.num_wires = num_wires
        self.required_params = 9 * num_wires - 4  # Total number of required parameters

    def apply(self, v):
        """
        Applies the quantum neural network layer with the given parameters.

        Parameters:
        - v (list or array): List or array of parameters for the quantum gates.

        For each layer, the number of parameters to train is 9 * num_wires - 4
        - 2 Interferometers: 6*num_wires - 4
          - 2 Beamsplitters: 2*(2*(num_wires - 1))
          - 2 Rotation Gates: 2*num_wires
        - Squeezing Gate: num_wires
        - Displacement Gate: num_wires
        - Kerr Gate: num_wires
        Total: 9 * num_wires - 4
        """
        # Guardrail: Ensure v has the correct length
        if len(v) > self.required_params:
            print("The entries longer than 9*num_wires - 4 are discarded.")
            v = v[:self.required_params]  # Truncate excess parameters
        elif len(v) < self.required_params:
            print("The parameters you provided are not enough. The rest will be zero-padded to fit 9*num_wires - 4.")
            v = list(v) + [0.0] * (self.required_params - len(v))  # Zero-pad missing parameters

        num_params = len(v)
        param_idx = 0

        # Interferometer 1: Beamsplitters + Rotations
        for i in range(self.num_wires - 1):
            if param_idx + 1 < num_params:
                qml.Beamsplitter(v[param_idx], v[param_idx + 1], wires=[i, i + 1])
                param_idx += 2

        for i in range(self.num_wires):
            if param_idx < num_params:
                qml.Rotation(v[param_idx], wires=i)
                param_idx += 1

        # Squeezing Gates
        for i in range(self.num_wires):
            if param_idx < num_params:
                qml.Squeezing(v[param_idx], 0.0, wires=i)
                param_idx += 1

        # Interferometer 2: Beamsplitters + Rotations
        for i in range(self.num_wires - 1):
            if param_idx + 1 < num_params:
                qml.Beamsplitter(v[param_idx], v[param_idx + 1], wires=[i, i + 1])
                param_idx += 2

        for i in range(self.num_wires):
            if param_idx < num_params:
                qml.Rotation(v[param_idx], wires=i)
                param_idx += 1

        # Bias addition
        for i in range(self.num_wires):
            if param_idx < num_params:
                qml.Displacement(v[param_idx], 0.0, wires=i)
                param_idx += 1

        # Non-linear activation function
        for i in range(self.num_wires):
            if param_idx < num_params:
                qml.Kerr(v[param_idx], wires=i)
                param_idx += 1