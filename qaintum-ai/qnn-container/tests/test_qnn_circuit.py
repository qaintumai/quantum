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
import torch
import pennylane as qml

from qnn.layers.qnn_circuit import QuantumNeuralNetworkCircuit

class TestQuantumNeuralNetworkCircuit(unittest.TestCase):

    def setUp(self):
        """
        Initialize a QuantumNeuralNetworkCircuit instance for testing.
        """
        self.num_wires = 4
        self.cutoff_dim = 5

    def test_single_output(self):
        """
        Test the circuit with output_size="single".
        """
        qnn_circuit = QuantumNeuralNetworkCircuit(num_wires=self.num_wires, cutoff_dim=self.cutoff_dim, output_size="single")
        circuit = qnn_circuit.build_circuit()

        inputs = torch.randn(8 * self.num_wires - 2).tolist()  # Convert to Python list
        var = [torch.randn(9 * self.num_wires - 4).tolist()]   # Convert to Python list
        output = circuit(inputs, var)

        # Ensure the output is a scalar tensor
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.dim(), 0)  # Check that it's a scalar tensor

    def test_multi_output(self):
        """
        Test the circuit with output_size="multi".
        """
        qnn_circuit = QuantumNeuralNetworkCircuit(num_wires=self.num_wires, cutoff_dim=self.cutoff_dim, output_size="multi")
        circuit = qnn_circuit.build_circuit()

        inputs = torch.randn(8 * self.num_wires - 2).tolist()  # Convert to Python list
        var = [torch.randn(9 * self.num_wires - 4).tolist()]   # Convert to Python list

        output = circuit(inputs, var)

        # Ensure the output is a list of length num_wires
        self.assertEqual(len(output), self.num_wires)

    def test_probabilities_output(self):
        """
        Test the circuit with output_size="probabilities".
        """
        qnn_circuit = QuantumNeuralNetworkCircuit(num_wires=self.num_wires, cutoff_dim=self.cutoff_dim, output_size="probabilities")
        circuit = qnn_circuit.build_circuit()

        inputs = torch.randn(8 * self.num_wires - 2).tolist()  # Convert to Python list
        var = [torch.randn(9 * self.num_wires - 4).tolist()]   # Convert to Python list

        output = circuit(inputs, var)

        # Ensure the output size is equal to cutoff_dim**num_wires
        self.assertEqual(len(output), self.cutoff_dim**self.num_wires)

    def test_invalid_output_size(self):
        """
        Test that the circuit raises an error for invalid output_size values.
        """
        with self.assertRaises(ValueError):
            QuantumNeuralNetworkCircuit(num_wires=self.num_wires, cutoff_dim=self.cutoff_dim, output_size="invalid")

if __name__ == '__main__':
    unittest.main()