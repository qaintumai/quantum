#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_encoding.py

Unit tests for the data_encoding function.
"""

import unittest
import numpy as np
import pennylane as qml
from src.models.qnn_layer import data_encoding

class TestDataEncoding(unittest.TestCase):
    
    def setUp(self):
        self.num_wires = 8
        self.num_basis = 2  # Example cutoff dimension
        self.num_features = self.num_wires * 10  # Example: Number of features should be more than twice the number of wires
        self.x = np.random.rand(self.num_features)

    def test_data_encoding_execution(self):
        """Test if the data_encoding function executes without errors"""
        try:
            dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=self.num_basis)
            @qml.qnode(dev)
            def circuit(x):
                data_encoding(x, num_wires=self.num_wires)
                return [qml.expval(qml.X(w)) for w in range(self.num_wires)]
            circuit(self.x)
        except Exception as e:
            self.fail(f"data_encoding raised an exception: {e}")

    def test_data_encoding_shapes(self):
        """Test if the data_encoding function handles different shapes of input correctly"""
        for features in [self.num_wires * 2, self.num_wires * 5, self.num_wires * 10]:
            x = np.random.rand(features)
            try:
                dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=self.num_basis)
                @qml.qnode(dev)
                def circuit(x):
                    data_encoding(x, num_wires=self.num_wires)
                    return [qml.expval(qml.X(w)) for w in range(self.num_wires)]
                circuit(x)
            except Exception as e:
                self.fail(f"data_encoding failed for input shape {x.shape}: {e}")

if __name__ == '__main__':
    unittest.main()
