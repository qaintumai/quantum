#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_init_weights.py

Unit tests for the init_weights function in the QuantumNeuralNetwork class.
"""

import unittest
import numpy as np
from quantum_neural_networks.src.qnn_model import QuantumNeuralNetwork

class TestInitWeights(unittest.TestCase):
    
    def setUp(self):
        self.qnn = QuantumNeuralNetwork(num_wires=8)
        self.layers = 3
        self.modes = self.qnn.num_wires

    def test_init_weights_shape(self):
        """Test if the init_weights method returns weights of correct shape"""
        weights = self.qnn.init_weights(self.layers, self.modes)
        expected_shape = (self.layers, (self.modes - 1) * 2 + self.modes + self.modes + (self.modes - 1) * 2 + self.modes + self.modes)
        self.assertEqual(weights.shape, expected_shape, f"Expected shape {expected_shape}, but got {weights.shape}")

    def test_init_weights_values(self):
        """Test if the init_weights method returns weights with appropriate values"""
        active_sd = 0.0001
        passive_sd = 0.1
        weights = self.qnn.init_weights(self.layers, self.modes, active_sd=active_sd, passive_sd=passive_sd)

        # Extract different parts of the weights
        M = (self.modes - 1) * 2 + self.modes
        int1_weights = weights[:, :M]
        s_weights = weights[:, M:M + self.modes]
        int2_weights = weights[:, M + self.modes:M + self.modes + M]
        dr_weights = weights[:, M + self.modes + M:M + self.modes + M + self.modes]
        k_weights = weights[:, M + self.modes + M + self.modes:]

        # Check the standard deviations
        self.assertAlmostEqual(np.std(int1_weights), passive_sd, delta=0.01, msg="Interferometer 1 weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(s_weights), active_sd, delta=0.01, msg="Squeezing weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(int2_weights), passive_sd, delta=0.01, msg="Interferometer 2 weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(dr_weights), active_sd, delta=0.01, msg="Displacement weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(k_weights), active_sd, delta=0.01, msg="Kerr weights standard deviation out of bounds")

if __name__ == '__main__':
    unittest.main()

