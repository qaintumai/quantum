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
import torch
import numpy as np

from qnn.layers.qnn_circuit import QuantumNeuralNetworkCircuit
from qnn.utils.weight_initializer import WeightInitializer  # Import WeightInitializer

class QuantumNeuralNetwork:
    def __init__(self, num_wires=4, cutoff_dim=5, num_layers=2, output_size="single"):
        """
        Initializes the quantum neural network model.

        Parameters:
        - num_wires (int): Number of quantum wires (qumodes) in the circuit.
        - cutoff_dim (int): Cutoff dimension for the Fock space.
        - num_layers (int): Number of quantum layers in the circuit.
        - output_size (str): Type of output ("single", "multi", or "probabilities").
                             Defaults to "single".
        """
        self.num_wires = num_wires
        self.cutoff_dim = cutoff_dim
        self.num_layers = num_layers
        self.output_size = output_size.lower()

        # Validate output_size
        if self.output_size not in ["single", "multi", "probabilities"]:
            raise ValueError("output_size must be one of 'single', 'multi', or 'probabilities'.")

        # Initialize the quantum circuit
        self.qnn_circuit = QuantumNeuralNetworkCircuit(
            num_wires=self.num_wires,
            cutoff_dim=self.cutoff_dim,
            output_size=self.output_size
        ).build_circuit()

        # Initialize weights for quantum layers using WeightInitializer
        self.weights = self._initialize_weights()

        # Convert the quantum layer to a Torch layer
        self.qlayers = self._build_quantum_layers()

    def _initialize_weights(self):
        """
        Initializes trainable weights for the quantum layers using WeightInitializer.

        Returns:
            torch.Tensor: Randomly initialized weights for the quantum layers.
        """
        # Use WeightInitializer to generate weights
        weights_np = WeightInitializer.init_weights(
            layers=self.num_layers,
            num_wires=self.num_wires,
            active_sd=0.0001,
            passive_sd=0.1
        )

        # Convert the numpy array to a PyTorch tensor
        return torch.tensor(weights_np, dtype=torch.float32)

    def _build_quantum_layers(self):
        """
        Converts the quantum neural network to a Torch layer.

        Returns:
            qml.qnn.TorchLayer: A Torch-compatible quantum layer.
        """
        # Define the shape of the weights
        weight_shapes = {"var": (self.num_layers, 9 * self.num_wires - 4)}

        # Create a TorchLayer from the quantum circuit
        return qml.qnn.TorchLayer(self.qnn_circuit, weight_shapes)

    def forward(self, inputs):
        """
        Performs a forward pass through the quantum neural network.

        Parameters:
            inputs (torch.Tensor): Input data to encode into the quantum state.

        Returns:
            torch.Tensor: Output of the quantum circuit.
        """
        # Ensure inputs are a PyTorch tensor
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Inputs must be a PyTorch tensor.")

        # Convert the tensor to a Python list for compatibility with the encoder
        #inputs = inputs.tolist()

        # Pass the inputs through the quantum layers
        return self.qlayers(inputs)