#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnn_model.py

Quantum neural network model implementation using Pennylane and PyTorch.
"""

import torch
import numpy as np
import pennylane as qml

# Think through the output
num_modes = 8
num_basis = 2

# select a device
dev = qml.device("strawberryfields.fock", wires=num_modes, cutoff_dim=num_basis)

# Initialize the Quantum Neural Network
qnn = QuantumNeuralNetwork(num_wires=num_modes)

@qml.qnode(dev, interface="torch")
def quantum_nn(inputs, var):
    qnn.data_encoding(inputs)

    # iterative quantum layers
    for v in var:
        qnn.qnn_layer(v)

    return qml.expval(qml.X(0))

num_layers = 2

# initialize weights for quantum layers
weights = qnn.init_weights(num_layers, num_modes)

# convert the quantum layer to a PyTorch layer
shape_tup = weights.shape
weight_shapes = {'var': shape_tup}

qlayer = qml.qnn.TorchLayer(quantum_nn, weight_shapes)
layers = [qlayer]
model = torch.nn.Sequential(*layers)
