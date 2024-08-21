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

# This is based on Digital Quantum Computing. This needs to be modified to an Analog QC version.
import unittest
import pennylane as qml
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from layers.quantum_data_encoder import QuantumDataEncoder

class TestQuantumDataEncoder(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment with a PennyLane quantum device and an instance
        of the QuantumDataEncoder class.
        """
        self.num_wires = 4  # Example number of wires
        self.encoder = QuantumDataEncoder(num_wires=self.num_wires)

        # Use PennyLane's default.qubit simulator for testing
        # self.dev = qml.device("default.qubit", wires=self.num_wires)
        self.dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=2)


    def test_encoding_applies_gates(self):
        """
        Test that the QuantumDataEncoder applies the expected quantum gates.
        """
        # Create example input data
        num_params = 8 * self.num_wires - 2
        input_data = torch.randn(num_params)  # Random input data for the encoder

        @qml.qnode(self.dev)
        def circuit(input_data):
            self.encoder.encode(input_data)
            # return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]
            return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]

        # Run the circuit
        output = circuit(input_data)

        # Ensure the circuit ran successfully and returns the expected number of outputs
        self.assertEqual(len(output), self.num_wires)

    def test_encoder_with_insufficient_data(self):
        """
        Test that the QuantumDataEncoder handles cases where there is insufficient data.
        """
        # Insufficient data, less than the required number of parameters
        insufficient_data = torch.randn(8 * self.num_wires - 4)

        @qml.qnode(self.dev)
        def circuit(insufficient_data):
            self.encoder.encode(insufficient_data)
            # return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]
            wires = list(range(self.num_wires))
            return [qml.probs(wires=wires)]

        # Run the circuit and ensure no errors are raised
        output = circuit(insufficient_data)
        self.assertEqual(len(output), self.num_wires)

    def test_encoder_with_exact_data(self):
        """
        Test that the QuantumDataEncoder works correctly when the number of features is exactly divisible.
        """
        # Exactly enough data for one round of encoding
        exact_data = torch.randn(8 * self.num_wires - 2)

        @qml.qnode(self.dev)
        def circuit(exact_data):
            self.encoder.encode(exact_data)
            # return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]
            wires = list(range(self.num_wires))
            return [qml.probs(wires=wires)]

        output = circuit(exact_data)
        self.assertEqual(len(output), self.num_wires)

    def test_encoder_with_multiple_rounds(self):
        """
        Test that the QuantumDataEncoder can handle multiple rounds of encoding.
        """
        # Multiple rounds of encoding
        multiple_rounds_data = torch.randn((8 * self.num_wires - 2) * 2)  # Enough data for two rounds

        @qml.qnode(self.dev)
        def circuit(multiple_rounds_data):
            self.encoder.encode(multiple_rounds_data)
            # return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]
            wires = list(range(self.num_wires))
            return [qml.probs(wires=wires)]

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
                # return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]
                wires = list(range(self.num_wires))
                return [qml.probs(wires=wires)]

            circuit(invalid_data)

if __name__ == '__main__':
    unittest.main()
