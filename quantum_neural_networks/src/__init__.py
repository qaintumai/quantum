"""
Quantum Neural Networks (QNN) package.

This package provides functionalities to create and train quantum neural networks.
"""

from .models.qnn_model import QuantumNeuralNetwork
from .utils.qnn_utils import prepare_data, evaluate_model

__all__ = [
    'QuantumNeuralNetwork',
    'prepare_data',
    'evaluate_model',
]
