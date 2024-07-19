"""
This module initializes and defines the public API for the layers package. The package contains various classes and functions used for the quantum transfomer and QNN, including embedding layers, attention mechanisms, quantum neural networks, and utility functions for initializing weights and encoding data. This API is intended to allow users to design their own Quantum learning models using the libraries below.

Usage:
To import the entire API from layers:
    from layers import *
"""


from .input_embedding import InputEmbedding
from .multi_headed_attention import MultiHeadedAttention
from .qnn_multi_output import qnn_multi_output
from .qnn_probabilities import qnn_probabilities
from .qnn_single_output import qnn_single_output
from .quantum_data_encoder import QuantumDataEncoder
from .quantum_feed_forward import QuantumFeedForward
from .quantum_layer import QuantumNeuralNetworkLayer
from .scaled_dot_product import ScaledDotProduct
from .weight_initializer import WeightInitializer

__all__ = [
    "InputEmbedding",
    "MultiHeadedAttention",
    "qnn_multi_output",
    "qnn_probabilities",
    "qnn_single_output",
    "QuantumDataEncoder",
    "QuantumFeedForward",
    "QuantumLayer",
    "ScaledDotProduct",
    "WeightInitializer",
]
