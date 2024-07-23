# Copyright 2015 The qAIntum.ai Authors. All Rights Reserved.
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
This module initializes and defines the public API for the layers package. The package
contains various classes and functions used for the quantum transfomer and QNN, including
embedding layers, attention mechanisms, quantum neural networks, and utility functions
for initializing weights and encoding data. This API is intended to allow users to design
their own Quantum learning models using the libraries below.

Usage:
To import the entire API from layers:
    from layers import *
"""


from .input_embedding import InputEmbedding
from .multi_headed_attention import MultiHeadedAttention
from .qnn_circuit import qnn_circuit
from .quantum_data_encoder import QuantumDataEncoder
from .quantum_feed_forward import QuantumFeedForward
from .quantum_layer import QuantumNeuralNetworkLayer
from .scaled_dot_product import ScaledDotProduct
from .weight_initializer import WeightInitializer

__all__ = [
    "InputEmbedding",
    "MultiHeadedAttention",
    "QuantumDataEncoder",
    "QuantumFeedForward",
    "QuantumNeuralNetworkLayer",
    "ScaledDotProduct",
    "WeightInitializer",
]
