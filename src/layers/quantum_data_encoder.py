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

class QuantumDataEncoder:
    """
    This module defines the QuantumDataEncoder class, which is used to encode classical data
    into a quantum format suitable for use in a quantum neural network. The class applies a
    series of quantum gates (squeezing, beamsplitter, rotation, displacement, and Kerr gates)
    to the input data.

    Usage:
    To use the QuantumDataEncoder class, import it as follows:
    from layers.quantum_data_encoder import QuantumDataEncoder

    Example:
    encoder = QuantumDataEncoder(num_wires=8)
    encoder.encode(input_data)
    """

    def __init__(self, num_wires):
        """
        Initializes the QuantumDataEncoder class with the given number of wires.

        Parameters:
        - num_wires (int): Number of quantum wires.

        NOTE: Currently, Xanadu's QPU X8 supports only up to 8 parameters.
              We checked and Pennylane supports more than 8 parameters.
        """
        self.num_wires = num_wires

    def encode(self, x):
        """
        Encodes the input data into a quantum state to be operated on using a sequence of quantum gates.

        Parameters:
        x : input data (list or array-like)

        The encoding process uses the following gates in sequence:
        - Squeezing gates: 2*self.num_wires parameters
        - Beamsplitter gates: 2(self.num_wires-1) parameters
        - Rotation gates: self.num_wires parameters
        - Displacement gates: 2*self.num_wires parameters
        - Kerr gates: self.num_wires parameters
          Total: 8*self.num_wires - 2 parameters

        rounds: the number of iterations of the sequence needed to take in all the entries of the input data
                num_features // (8 * self.num_wires - 2)
                We are adding (8 * self.num_wires - 3) as a pad to run one extra round for the remainding data entries.
        """
        num_features = len(x)

        # Calculate the number of rounds needed to process all features
        rounds = (num_features + (8 * self.num_wires - 3)) // (8 * self.num_wires - 2)

        for j in range(rounds):
            start_idx = j * (8 * self.num_wires - 2)

            # Squeezing gates
            for i in range(self.num_wires):
                # for each wire, the number of parameters are i*2
                idx = start_idx + i * 2
                if idx + 1 < num_features:
                    qml.Squeezing(x[idx], x[idx + 1], wires=i)

            # Beamsplitter gates
            for i in range(self.num_wires - 1):
                # start_index + Squeezing gates, and then i*2 parameters for each gate
                idx = start_idx + self.num_wires * 2 + i * 2
                if idx + 1 < num_features:
                    qml.Beamsplitter(x[idx], x[idx + 1], wires=[i % self.num_wires, (i + 1) % self.num_wires])

            # Rotation gates
            for i in range(self.num_wires):
                # start_index + Squeezing gates + Beamsplitters, and then i parameters for each gate
                idx = start_idx + self.num_wires * 2 + (self.num_wires - 1) * 2 + i
                if idx < num_features:
                    qml.Rotation(x[idx], wires=i)

            # Displacement gates
            for i in range(self.num_wires):
                # start_index + Squeezing gates + Beamsplitters + Rotation gates, and then i*2 parameters for each gate
                idx = start_idx + self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + i * 2
                if idx + 1 < num_features:
                    qml.Displacement(x[idx], x[idx + 1], wires=i)

            # Kerr gates
            for i in range(self.num_wires):
                # start_index + Squeezing gates + Beamsplitters + Rotation gates + Displacement gates, and then i parameters for each gate
                idx = start_idx + self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + i
                if idx < num_features:
                    qml.Kerr(x[idx], wires=i)