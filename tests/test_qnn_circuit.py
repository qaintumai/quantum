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
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from layers.qnn_circuit import qnn_circuit
from layers.quantum_data_encoder import QuantumDataEncoder
from layers.qnn_layer import QuantumNeuralNetworkLayer
from utils.config import num_wires, num_basis, single_output, multi_output, probabilities


class TestQNNCircuit(unittest.TestCase):

    def setUp(self):
        """
        Initialize a basic setup for QNN circuit testing.
        """
        self.num_wires = num_wires
        self.num_basis = num_basis

        # Create mock inputs and variables for the QNN circuit
        self.inputs = torch.tensor([0.5] * self.num_wires)
        self.var = [torch.tensor([0.1] * self.num_wires) for _ in range(3)]  # example variables

    def test_single_output(self):
        """
        Test that the QNN circuit returns a single output when configured for single output.
        """
        if single_output:
            output = qnn_circuit(self.inputs, self.var)
            self.assertIsInstance(output, float,
                                  "Expected a single output of type float when single_output is True")
            self.assertGreaterEqual(output, -1.0)
            self.assertLessEqual(output, 1.0)

    def test_multi_output(self):
        """
        Test that the QNN circuit returns multiple outputs when configured for multi-output.
        """
        if multi_output:
            output = qnn_circuit(self.inputs, self.var)
            self.assertIsInstance(output, list,
                                  "Expected a list of outputs when multi_output is True")
            self.assertEqual(len(output), self.num_wires,
                             f"Expected {self.num_wires} outputs, but got {len(output)}")
            for out in output:
                self.assertGreaterEqual(out, -1.0)
                self.assertLessEqual(out, 1.0)

    def test_probabilities_output(self):
        """
        Test that the QNN circuit returns probabilities when configured for probability output.
        """
        if probabilities:
            output = qnn_circuit(self.inputs, self.var)
            self.assertIsInstance(output, list,
                                  "Expected a list of probabilities when probabilities is True")
            self.assertEqual(len(output[0]), num_basis ** self.num_wires,
                             f"Expected {num_basis ** self.num_wires} probabilities, but got {len(output[0])}")
            self.assertTrue(all(0 <= prob <= 1 for prob in output[0]),
                            "All probabilities should be between 0 and 1.")
            self.assertAlmostEqual(sum(output[0]), 1.0,
                                   "The sum of the probabilities should be approximately 1.")

if __name__ == '__main__':
    # Run the test suite
    result = unittest.main(exit=False)

    # Check if all tests passed
    if result.result.wasSuccessful():
        print("Tests passed!")
