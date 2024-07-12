#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_init_weights.py

Unit tests for the init_weights function.
"""

import unittest
import numpy as np
from src.models.qnn_layer import init_weights

class TestInitWeights(unittest.TestCase):
    
    def setUp(self):
        self.num_wires = 8
        self.layers = 3

    def test_init_weights_shape(self):
        """Test if the init_weights function returns weights of correct shape"""
        weights = init_weights(self.layers, self.num_wires)
        expected_shape = (self.layers, (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires)
        self.assertEqual(weights.shape, expected_shape, f"Expected shape {expected_shape}, but got {weights.shape}")

    def test_init_weights_values(self):
        """Test if the init_weights function returns weights with appropriate values"""
        active_sd = 0.0001
        passive_sd = 0.1
        weights = init_weights(self.layers, self.num_wires, active_sd=active_sd, passive_sd=passive_sd)

        # Extract different parts of the weights
        M = (self.num_wires - 1) * 2 + self.num_wires
        int1_weights = weights[:, :M]
        s_weights = weights[:, M:M + self.num_wires]
        int2_weights = weights[:, M + self.num_wires:M + self.num_wires + M]
        dr_weights = weights[:, M + self.num_wires + M:M + self.num_wires + M + self.num_wires]
        k_weights = weights[:, M + self.num_wires + M + self.num_wires:]

        # Check the standard deviations
        self.assertAlmostEqual(np.std(int1_weights), passive_sd, delta=0.01, msg="Interferometer 1 weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(s_weights), active_sd, delta=0.01, msg="Squeezing weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(int2_weights), passive_sd, delta=0.01, msg="Interferometer 2 weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(dr_weights), active_sd, delta=0.01, msg="Displacement weights standard deviation out of bounds")
        self.assertAlmostEqual(np.std(k_weights), active_sd, delta=0.01, msg="Kerr weights standard deviation out of bounds")

if __name__ == '__main__':
    unittest.main()
