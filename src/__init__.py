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
This module initializes the primary components of the qAIntum.ai library.

It imports and exposes key classes and functions from the layers and models
submodules. These components are essential for building and working with
quantum neural networks and quantum transformers.

Available components:
- Layers:
    * InputEmbedding
    * MultiHeadedAttention
    * QuantumDataEncoder
    * QuantumNeuralNetworkLayer
    * ScaledDotProduct
    * WeightInitializer
    * qnn_circuit (Quantum Neural Network Circuit)
    
- Models:
    * QuantumDecoder
    * QuantumEncoder
    * QuantumFeedForward
    * QuantumNeuralNetwork
    * QuantumTransformer
"""


from .layers import (
    InputEmbedding,
    MultiHeadedAttention,
    QuantumDataEncoder,
    QuantumNeuralNetworkLayer,
    ScaledDotProduct,
    WeightInitializer,
)

from .models import (
    QuantumDecoder,
    QuantumEncoder,
    QuantumFeedForward,
    QuantumNeuralNetwork,
    QuantumTransformer,
)

__all__ = [
    "InputEmbedding",
    "MultiHeadedAttention",
    "qnn_circuit",
    "QuantumDataEncoder",
    "QuantumNeuralNetworkLayer",
    "ScaledDotProduct",
    "WeightInitializer",
    "QuantumDecoder",
    "QuantumEncoder",
    "QuantumFeedForward",
    "QuantumNeuralNetwork",
    "QuantumTransformer",
]
