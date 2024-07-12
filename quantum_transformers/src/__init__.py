"""
Package for Quantum Transformers (QT).

This package includes various layer and model implementations for QTs.
"""

# Import layers
from .layers import (
    InputEmbedding,
    MultiHeadedAttention,
    QuantumDataEncoding,
    QuantumFeedForward,
    ScaledDotProductAttention,
    WeightInitializer
)

# Import models
from .models import (
    QuantumDecoder,
    QuantumEncoder,
    QuantumTransformer
)

__all__ = [
    # Layers
    'InputEmbedding',
    'MultiHeadedAttention',
    'QuantumDataEncoding',
    'QuantumFeedForward',
    'ScaledDotProductAttention',
    'WeightInitializer',
    # Models
    'QuantumDecoder',
    'QuantumEncoder',
    'QuantumTransformer'
]
