#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_qnn_layer.py

Unit tests for the qnn_layer function.
"""

import unittest
import numpy as np
import pennylane as qml
from src.models.qnn_layer import qnn_layer

class TestQnnLayer(unittest.TestCase):
    
    def setUp(self):
        self.num_wires = 8
        self.num_basis = 2  # Example cutoff dimension
        self.num_params = self.num_wires * 10  # Example: Number of parameters should be more than required
        self.v = np.random.rand(self.num_params)

    def test_qnn_layer_execution(self):
        """Test if the qnn_layer function executes without errors"""
        try:
            dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=self.num_basis)
            @qml.qnode(dev)
            def circuit(v):
                qnn_layer(v, num_wires=self.num_wires)
                return [qml.expval(qml.X(w)) for w in range(self.num_wires)]
            circuit(self.v)
        except Exception as e:
            self.fail(f"qnn_layer raised an exception: {e}")

    def test_qnn_layer_shapes(self):
        """Test if the qnn_layer function handles different shapes of input correctly"""
        for params in [self.num_wires * 5, self.num_wires * 8, self.num_wires * 10]:
            v = np.random.rand(params)
            try:
                dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=self.num_basis)
                @qml.qnode(dev)
                def circuit(v):
                    qnn_layer(v, num_wires=self.num_wires)
                    return [qml.expval(qml.X(w)) for w in range(self.num_wires)]
                circuit(v)
            except Exception as e:
                self.fail(f"qnn_layer failed for input shape {v.shape}: {e}")

if __name__ == '__main__':
    unittest.main()
