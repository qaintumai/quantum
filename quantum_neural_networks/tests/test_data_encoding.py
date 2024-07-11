#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_encoding.py

Unit tests for the data_encoding function in the QuantumNeuralNetwork class.
"""

import unittest
import numpy as np
import pennylane as qml
from quantum_neural_networks.src.qnn_model import QuantumNeuralNetwork

class TestDataEncoding(unittest.TestCase):
    
    def setUp(self):
        self.qnn = QuantumNeuralNetwork(num_wires=8)
        self.num_features = self.qnn.num_wires * 10  # Example: Number of features should be more than twice the number of wires
        self.x = np.random.rand(self.num_features)

    def test_data_encoding_execution(self):
        """Test if the data_encoding method executes without errors"""
        try:
            dev = qml.device('default.qubit', wires=self.qnn.num_wires)
            @qml.qnode(dev)
            def circuit(x):
                self.qnn.data_encoding(x)
                return [qml.expval(qml.PauliZ(w)) for w in range(self.qnn.num_wires)]
            circuit(self.x)
        except Exception as e:
            self.fail(f"data_encoding raised an exception: {e}")

    def test_data_encoding_shapes(self):
        """Test if the data_encoding method handles different shapes of input correctly"""
        for features in [self.qnn.num_wires * 2, self.qnn.num_wires * 5, self.qnn.num_wires * 10]:
            x = np.random.rand(features)
            try:
                dev = qml.device('default.qubit', wires=self.qnn.num_wires)
                @qml.qnode(dev)
                def circuit(x):
                    self.qnn.data_encoding(x)
                    return [qml.expval(qml.PauliZ(w)) for w in range(self.qnn.num_wires)]
                circuit(x)
            except Exception as e:
                self.fail(f"data_encoding failed for input shape {x.shape}: {e}")

if __name__ == '__main__':
    unittest.main()

