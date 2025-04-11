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

import unittest
import pennylane as qml
import torch

from qnn.layers.qnn_layer import QuantumNeuralNetworkLayer

class TestQuantumNeuralNetworkLayer(unittest.TestCase):

    def setUp(self):
        """
        Initialize a quantum device and a QuantumNeuralNetworkLayer instance for testing.
        """
        self.num_wires = 4  # Example number of wires
        self.required_params = 9 * self.num_wires - 4  # Total number of required parameters
        self.qnn_layer = QuantumNeuralNetworkLayer(num_wires=self.num_wires)

        # Use Strawberry Fields' Fock device for continuous-variable quantum computing
        self.dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=2)

    def test_layer_applies_correct_operations(self):
        """
        Test that the QuantumNeuralNetworkLayer applies the correct operations based on the parameters.
        """
        params = torch.tensor([0.1] * self.required_params)  # Exact number of required parameters

        @qml.qnode(self.dev)
        def circuit(params):
            self.qnn_layer.apply(params)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output = circuit(params)

        # Assert that the circuit ran successfully and returns an output of expected size
        self.assertEqual(len(output), self.num_wires)

    def test_circuit_with_shorter_params(self):
        """
        Test that the QuantumNeuralNetworkLayer handles shorter parameter lists by zero-padding.
        """
        short_params = torch.tensor([0.1] * (self.required_params - 5))  # Shorter than required

        @qml.qnode(self.dev)
        def circuit(params):
            self.qnn_layer.apply(params)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output = circuit(short_params)

        # Ensure the circuit runs successfully despite the shorter input
        self.assertEqual(len(output), self.num_wires)

    def test_circuit_with_longer_params(self):
        """
        Test that the QuantumNeuralNetworkLayer handles longer parameter lists by truncation.
        """
        long_params = torch.tensor([0.1] * (self.required_params + 5))  # Longer than required

        @qml.qnode(self.dev)
        def circuit(params):
            self.qnn_layer.apply(params)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output = circuit(long_params)

        # Ensure the circuit runs successfully despite the longer input
        self.assertEqual(len(output), self.num_wires)

    def test_apply_edge_case_params(self):
        """
        Test the application of edge-case parameters, such as zeros or extreme values.
        """
        zero_params = torch.zeros(self.required_params)  # All parameters are zero
        extreme_params = torch.tensor([100.0] * self.required_params)  # Extreme parameter values

        @qml.qnode(self.dev)
        def circuit(params):
            self.qnn_layer.apply(params)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output_zero = circuit(zero_params)
        output_extreme = circuit(extreme_params)

        # Ensure the circuit runs without errors
        self.assertEqual(len(output_zero), self.num_wires)
        self.assertEqual(len(output_extreme), self.num_wires)

if __name__ == '__main__':
    unittest.main()