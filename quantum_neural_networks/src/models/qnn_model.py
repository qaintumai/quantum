#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnn.py

Simple hybrid quantum neural network developed in Pennylane (quantum) and PyTorch (classical).
"""

import numpy as np
import pennylane as qml

class QuantumNeuralNetwork:
    def __init__(self, num_wires=8):
        self.num_wires = num_wires

    def data_encoding(self, x):
        """
        Encode classical data into quantum states using a series of quantum gates.

        Parameters:
        x (array-like): Input data to be encoded.
        """
        num_features = len(x)

        # Squeezing gates
        for i in range(0, min(num_features, self.num_wires * 2), 2):
            qml.Squeezing(x[i], x[i + 1], wires=i // 2)

        # Beamsplitter gates
        for i in range(self.num_wires - 1):
            idx = self.num_wires * 2 + i * 2
            if idx + 1 < num_features:
                qml.Beamsplitter(x[idx], x[idx + 1], wires=[i % self.num_wires, (i + 1) % self.num_wires])

        # Rotation gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + i
            if idx < num_features:
                qml.Rotation(x[idx], wires=i)

        # Displacement gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + i * 2
            if idx + 1 < num_features:
                qml.Displacement(x[idx], x[idx + 1], wires=i)

        # Kerr gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + i
            if idx < num_features:
                qml.Kerr(x[idx], wires=i)

        # Squeezing gates (second set)
        for i in range(0, min(num_features - (self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires), self.num_wires * 2), 2):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires + i
            if idx + 1 < num_features:
                qml.Squeezing(x[idx], x[idx + 1], wires=i // 2)

        # Rotation gates (second set)
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires + self.num_wires * 2 + i
            if idx < num_features:
                qml.Rotation(x[idx], wires=i)

    def qnn_layer(self, v):
        """
        Define a quantum neural network layer using a series of quantum gates.

        Parameters:
        v (array-like): Parameters for the quantum gates.
        """
        num_params = len(v)

        # Interferometer 1
        for i in range(self.num_wires - 1):
            idx = i * 2
            if idx + 1 < num_params:
                theta = v[idx]
                phi = v[idx + 1]
                qml.Beamsplitter(theta, phi, wires=[i % self.num_wires, (i + 1) % self.num_wires])

        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + i
            if idx < num_params:
                qml.Rotation(v[idx], wires=i)

        # Squeezers
        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + i
            if idx < num_params:
                qml.Squeezing(v[idx], 0.0, wires=i)

        # Interferometer 2
        for i in range(self.num_wires - 1):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + i * 2
            if idx + 1 < num_params:
                theta = v[idx]
                phi = v[idx + 1]
                qml.Beamsplitter(theta, phi, wires=[i % self.num_wires, (i + 1) % self.num_wires])

        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + (self.num_wires - 1) * 2 + i
            if idx < num_params:
                qml.Rotation(v[idx], wires=i)

        # Bias addition
        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + (self.num_wires - 1) * 2 + self.num_wires + i
            if idx < num_params:
                qml.Displacement(v[idx], 0.0, wires=i)

        # Non-linear activation function
        for i in range(self.num_wires):
            idx = (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires + i
            if idx < num_params:
                qml.Kerr(v[idx], wires=i)

    def init_weights(self, layers, modes, active_sd=0.0001, passive_sd=0.1):
        """
        Initialize weights for the quantum neural network.

        Parameters:
        layers (int): Number of layers in the network.
        modes (int): Number of modes (wires) in the network.
        active_sd (float): Standard deviation for active gate weights.
        passive_sd (float): Standard deviation for passive gate weights.

        Returns:
        array: Initialized weights.
        """
        M = (modes - 1) * 2 + modes  # Number of interferometer parameters

        int1_weights = np.random.normal(size=[layers, M], scale=passive_sd)
        s_weights = np.random.normal(size=[layers, modes], scale=active_sd)
        int2_weights = np.random.normal(size=[layers, M], scale=passive_sd)
        dr_weights = np.random.normal(size=[layers, modes], scale=active_sd)
        k_weights = np.random.normal(size=[layers, modes], scale=active_sd)

        weights = np.concatenate([int1_weights, s_weights, int2_weights, dr_weights, k_weights], axis=1)

        return weights
