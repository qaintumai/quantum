import unittest
import torch
import pennylane as qml
from quantum.src.layers.qnn_circuit import qnn_circuit
from quantum.src.layers.quantum_data_encoder import QuantumDataEncoder
from quantum.src.layers.qnn_layer import QuantumNeuralNetworkLayer
from quantum.src.utils.config import num_wires, num_basis, single_output, multi_output, probabilities


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
            self.assertEqual(len(output[0]), 2 ** self.num_wires, 
                             f"Expected {2 ** self.num_wires} probabilities, but got {len(output[0])}")
            self.assertTrue(all(0 <= prob <= 1 for prob in output[0]), 
                            "All probabilities should be between 0 and 1.")
            self.assertAlmostEqual(sum(output[0]), 1.0, 
                                   "The sum of the probabilities should be approximately 1.")

    def test_invalid_inputs(self):
        """
        Test that the QNN circuit raises an error for invalid input dimensions.
        """
        with self.assertRaises(Exception):
            # Provide invalid input dimension
            invalid_inputs = torch.tensor([0.5] * (self.num_wires + 1))
            qnn_circuit(invalid_inputs, self.var)

    def test_invalid_variables(self):
        """
        Test that the QNN circuit raises an error for invalid variable dimensions.
        """
        with self.assertRaises(Exception):
            # Provide invalid variable dimensions
            invalid_var = [torch.tensor([0.1] * (self.num_wires + 1)) for _ in range(3)]
