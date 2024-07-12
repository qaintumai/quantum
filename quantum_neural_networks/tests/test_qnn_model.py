#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_get_model.py

Unit tests for the get_model function in qnn_model.py.
"""

import unittest
import torch
from src.models.qnn_model import get_model

class TestGetModel(unittest.TestCase):
    
    def setUp(self):
        self.num_wires = 8
        self.num_layers = 2

    def test_get_model_instance(self):
        """Test if the get_model function returns a PyTorch Sequential model instance"""
        try:
            model = get_model(self.num_wires, self.num_layers)
            self.assertIsInstance(model, torch.nn.Sequential)
        except Exception as e:
            self.fail(f"get_model raised an exception: {e}")

    def test_get_model_structure(self):
        """Test if the get_model function returns a model with the correct structure"""
        try:
            model = get_model(self.num_wires, self.num_layers)
            # Ensure the model has the expected layer type
            self.assertTrue(any(isinstance(layer, qml.qnn.TorchLayer) for layer in model))
        except Exception as e:
            self.fail(f"get_model raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
