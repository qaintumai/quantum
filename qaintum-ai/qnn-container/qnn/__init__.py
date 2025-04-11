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
This module initializes and defines the public API for the QNN package. The package
contains various classes and functions used for quantum neural networks, including
subdirectories: layers, models, and utils.

This API provides an interface for designing Quantum Neural Networks (QNNs) by exposing
essential components from these submodules.

Usage:
To import the entire API from layers:
    from qnn.layers import *

To import specific components:
    from qnn import QuantumNeuralNetwork, WeightInitializer
"""

from .models import QuantumNeuralNetwork
from .layers import *
from .utils import WeightInitializer

__all__ = [
           "QuantumNeuralNetwork",
           "WeightInitializer",
          ]

