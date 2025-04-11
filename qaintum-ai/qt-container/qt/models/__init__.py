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
This module initializes and defines the public API for the models package in the Quantum Transformer (QT).
It includes core components required for constructing and training quantum-enhanced transformer models:

- **QuantumEncoder**: Encodes input data into a quantum-compatible representation.
- **QuantumDecoder**: Decodes quantum-encoded representations back into meaningful output.
- **QuantumFeedForward**: Implements a quantum-enhanced feedforward network for transformation.
- **QuantumTransformer**: The full-stack quantum transformer model integrating encoding, decoding, and attention.

These models provide the foundation for developing quantum-native and hybrid quantum-classical transformer architectures.

Usage:
To import all available models:
    from QT.models import *
"""

from .quantum_decoder import QuantumDecoder
from .quantum_encoder import QuantumEncoder
from .quantum_feed_forward import QuantumFeedForward
from .quantum_transformer import QuantumTransformer

__all__ = [
    "QuantumDecoder",
    "QuantumEncoder",
    "QuantumFeedForward",
    "QuantumTransformer",
]

