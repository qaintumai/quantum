# Copyright 2024 The qAIntum.ai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This module initializes and defines the public API for the layers package in the Quantum Transformer (QT).
It provides essential components for building quantum transformers, including:

- **InputEmbedding**: Converts input data into a quantum-compatible representation.
- **MultiHeadedAttention**: Implements a quantum-enhanced multi-head attention mechanism.
- **ScaledDotProduct**: Computes attention scores for efficient sequence learning.

These layers form the foundation for embedding and attention mechanisms in quantum transformers.
This API enables users to design and customize their own quantum learning models.

Usage:
To import all available layers:
    from QT.layers import *
"""

from .input_embedding import InputEmbedding
from .multi_headed_attention import MultiHeadedAttention
from .scaled_dot_product import ScaledDotProduct

__all__ = [
    "InputEmbedding",
    "MultiHeadedAttention",
    "ScaledDotProduct",
]

