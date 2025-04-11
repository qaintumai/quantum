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

try:
    from qnn.layers.qnn_data_encoder import QuantumDataEncoder
    print("Import successful!")
except ImportError as e:
    print("Import failed:", e)
    raise

class TestQuantumDataEncoder(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment with a PennyLane quantum device and an instance
        of the QuantumDataEncoder class.
        """
        self.num_wires = 4  # Example number of wires
        self.encoder = QuantumDataEncoder(num_wires=self.num_wires)

        # Use Fock basis for Continuous Variable Model of Quantum Computing
        self.dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=2)

    def test_encoding_applies_gates(self):
        """
        Test that the QuantumDataEncoder applies the expected quantum gates.
        """
        num_params = 8 * self.num_wires - 2
        input_data = torch.randn(num_params).tolist()  # Convert to Python list

        @qml.qnode(self.dev)
        def circuit(input_data):
            self.encoder.encode(input_data)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output = circuit(input_data)
        self.assertEqual(len(output), self.num_wires)

    def test_encoder_with_insufficient_data(self):
        """
        Test that the QuantumDataEncoder handles cases where there is insufficient data.
        """
        insufficient_data = torch.randn(2 * self.num_wires - 1).tolist()  # Convert to Python list

        @qml.qnode(self.dev)
        def circuit(insufficient_data):
            self.encoder.encode(insufficient_data)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output = circuit(insufficient_data)
        self.assertEqual(len(output), self.num_wires)

    def test_encoder_with_exact_data(self):
        """
        Test that the QuantumDataEncoder works correctly when the number of features is exactly divisible.
        """
        exact_data = torch.randn(8 * self.num_wires - 2).tolist()  # Convert to Python list

        @qml.qnode(self.dev)
        def circuit(exact_data):
            self.encoder.encode(exact_data)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output = circuit(exact_data)
        self.assertEqual(len(output), self.num_wires)

    def test_encoder_with_multiple_rounds(self):
        """
        Test that the QuantumDataEncoder can handle multiple rounds of encoding.
        """
        multiple_rounds_data = torch.randn((8 * self.num_wires - 2) * 2).tolist()  # Convert to Python list

        @qml.qnode(self.dev)
        def circuit(multiple_rounds_data):
            self.encoder.encode(multiple_rounds_data)
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        output = circuit(multiple_rounds_data)
        self.assertEqual(len(output), self.num_wires)

    def test_invalid_data_type(self):
        """
        Test that the QuantumDataEncoder raises an error when given invalid input data.
        """
        invalid_data = "invalid input data"  # Non-numeric input

        with self.assertRaises(TypeError):
            @qml.qnode(self.dev)
            def circuit(invalid_data):
                self.encoder.encode(invalid_data)
                return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

            circuit(invalid_data)

if __name__ == '__main__':
    unittest.main()