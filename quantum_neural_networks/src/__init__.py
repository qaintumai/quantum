"""
Quantum Neural Networks (QNN) package.

This package provides functionalities to create and train quantum neural networks.
"""

from .qnn_model import QuantumNeuralNetwork
from .qnn_utils import prepare_data, evaluate_model

__all__ = [
    'QuantumNeuralNetwork',
    'prepare_data',
    'evaluate_model',
]
