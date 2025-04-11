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

from qnn.models.quantum_neural_network import QuantumNeuralNetwork

class TestQuantumNeuralNetwork(unittest.TestCase):

    def setUp(self):
        """
        Initialize common parameters for testing.
        """
        self.num_wires = 2
        self.cutoff_dim = 2
        self.num_layers = 2

    def test_single_output(self):
        """
        Test the quantum neural network with output_size="single".
        """
        # Initialize the quantum neural network
        qnn_model = QuantumNeuralNetwork(
            num_wires=self.num_wires,
            cutoff_dim=self.cutoff_dim,
            num_layers=self.num_layers,
            output_size="single"
        )

        # Generate example input data
        inputs = torch.randn(8 * self.num_wires - 2)  # Example input size

        # Perform a forward pass
        output = qnn_model.forward(inputs)

        # Ensure the output is a scalar tensor
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.dim(), 0)  # Check that it's a scalar tensor

    def test_multi_output(self):
        """
        Test the quantum neural network with output_size="multi".
        """
        # Initialize the quantum neural network
        qnn_model = QuantumNeuralNetwork(
            num_wires=self.num_wires,
            cutoff_dim=self.cutoff_dim,
            num_layers=self.num_layers,
            output_size="multi"
        )

        # Generate example input data
        inputs = torch.randn(8 * self.num_wires - 2)  # Example input size

        # Perform a forward pass
        output = qnn_model.forward(inputs)

        # Ensure the output is a list of tensors with length equal to num_wires
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.size(0), self.num_wires)  # Check the number of outputs

    def test_probabilities_output(self):
        """
        Test the quantum neural network with output_size="probabilities".
        """
        # Initialize the quantum neural network
        qnn_model = QuantumNeuralNetwork(
            num_wires=self.num_wires,
            cutoff_dim=self.cutoff_dim,
            num_layers=self.num_layers,
            output_size="probabilities"
        )

        # Generate example input data
        inputs = torch.randn(8 * self.num_wires - 2)  # Example input size

        # Perform a forward pass
        output = qnn_model.forward(inputs)

        # Ensure the output size matches the total number of basis states
        expected_size = self.cutoff_dim ** self.num_wires
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.size(0), expected_size)  # Check the size of the probability distribution

        # Ensure the output is a valid probability distribution (sums to 1)
        self.assertAlmostEqual(output.sum().item(), 1.0, places=5)

    def test_invalid_output_size(self):
        """
        Test that the quantum neural network raises an error for invalid output_size values.
        """
        with self.assertRaises(ValueError):
            QuantumNeuralNetwork(
                num_wires=self.num_wires,
                cutoff_dim=self.cutoff_dim,
                num_layers=self.num_layers,
                output_size="invalid"
            )

    def test_forward_pass_with_custom_parameters(self):
        """
        Test the forward pass with custom num_wires, cutoff_dim, and num_layers.
        """
        # Custom parameters
        num_wires = 6
        cutoff_dim = 7
        num_layers = 4

        # Initialize the quantum neural network
        qnn_model = QuantumNeuralNetwork(
            num_wires=num_wires,
            cutoff_dim=cutoff_dim,
            num_layers=num_layers,
            output_size="multi"
        )

        # Generate example input data
        inputs = torch.randn(8 * num_wires - 2)  # Example input size

        # Perform a forward pass
        output = qnn_model.forward(inputs)

        # Ensure the output is a list of tensors with length equal to num_wires
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.size(0), num_wires)  # Check the number of outputs

if __name__ == '__main__':
    unittest.main()