#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_qnn_layer.py

Unit tests for the qnn_layer function in the QuantumNeuralNetwork class.
"""

import unittest
import numpy as np
import pennylane as qml
from quantum_neural_networks.src.qnn_model import QuantumNeuralNetwork

class TestQnnLayer(unittest.TestCase):
    
    def setUp(self):
        self.qnn = QuantumNeuralNetwork(num_wires=8)
        self.num_params = self.qnn.num_wires * 10  # Example: Number of parameters should be more than required
        self.v = np.random.rand(self.num_params)

    def test_qnn_layer_execution(self):
        """Test if the qnn_layer method executes without errors"""
        try:
            dev = qml.device('default.qubit', wires=self.qnn.num_wires)
            @qml.qnode(dev)
            def circuit(v):
                self.qnn.qnn_layer(v)
                return [qml.expval(qml.PauliZ(w)) for w in range(self.qnn.num_wires)]
            circuit(self.v)
        except Exception as e:
            self.fail(f"qnn_layer raised an exception: {e}")

    def test_qnn_layer_shapes(self):
        """Test if the qnn_layer method handles different shapes of input correctly"""
        for params in [self.qnn.num_wires * 5, self.qnn.num_wires * 8, self.qnn.num_wires * 10]:
            v = np.random.rand(params)
            try:
                dev = qml.device('default.qubit', wires=self.qnn.num_wires)
                @qml.qnode(dev)
                def circuit(v):
                    self.qnn.qnn_layer(v)
                    return [qml.expval(qml.PauliZ(w)) for w in range(self.qnn.num_wires)]
                circuit(v)
            except Exception as e:
                self.fail(f"qnn_layer failed for input shape {v.shape}: {e}")

if __name__ == '__main__':
    unittest.main()

