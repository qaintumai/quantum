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

import pennylane as qml
from itertools import zip_longest
import math

class QuantumDataEncoder:
    """
    Encodes classical data into quantum states using squeezing, beamsplitter,
    rotation, displacement, and Kerr gates.

    Usage:
        from layers.quantum_data_encoder import QuantumDataEncoder
        encoder = QuantumDataEncoder(num_wires=8)
        encoder.encode(input_data)
    """

    def __init__(self, num_wires):
        """
        Initializes the QuantumDataEncoder.

        Parameters:
            num_wires (int): Number of quantum wires.
        """
        self.num_wires = num_wires
        self.params_per_round = 8 * num_wires - 2  # Parameters required per encoding cycle

    def encode(self, x):
        """
        Encodes input data into quantum states.

        Parameters:
            x (list or array-like): Input data.

        Raises:
            ValueError: If the input size is too small relative to the number of wires.
        """
        # Validate the input size
        if len(x) <= 2 * (self.num_wires - 1):
            raise ValueError("Please lower the number of wires.")

        # Validate that all elements of x are numeric
        if not all(isinstance(val, (int, float)) for val in x):
            raise TypeError("All elements of the input data must be numeric.")

        rounds = math.ceil(len(x) / self.params_per_round)

        for j in range(rounds):
            start_idx = j * self.params_per_round
            params = x[start_idx:start_idx + self.params_per_round]

            # Apply Squeezing gates
            for i, (r, phi) in zip(range(self.num_wires), zip_longest(params[::2], params[1::2], fillvalue=0)):
                qml.Squeezing(r, phi, wires=i)

            # Apply Beamsplitter gates
            offset = 2 * self.num_wires
            for (theta, phi), (i, j) in zip(zip_longest(params[offset::2], params[offset+1::2], fillvalue=0), zip(range(self.num_wires - 1), range(1, self.num_wires))):
                qml.Beamsplitter(theta, phi, wires=[i, j])

            # Apply Rotation gates
            offset += 2 * (self.num_wires - 1)
            for i, theta in zip(range(self.num_wires), params[offset:offset + self.num_wires]):
                qml.Rotation(theta, wires=i)

            # Apply Displacement gates
            offset += self.num_wires
            for i, (alpha, phi) in zip(range(self.num_wires), zip_longest(params[offset::2], params[offset+1::2], fillvalue=0)):
                qml.Displacement(alpha, phi, wires=i)

            # Apply Kerr gates
            offset += 2 * self.num_wires
            for i, kappa in zip(range(self.num_wires), params[offset:offset + self.num_wires]):
                qml.Kerr(kappa, wires=i)
