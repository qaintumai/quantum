#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnn_model.py

Quantum neural network model implementation using Pennylane and PyTorch.
"""

import torch
import numpy as np
import pennylane as qml
from .qnn_layer import data_encoding, qnn_layer, init_weights

num_wires = 8
num_basis = 2

# select a device
dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=num_basis)

@qml.qnode(dev, interface="torch")
def quantum_nn(inputs, var):
    # convert classical inputs into quantum states
    data_encoding(inputs)

    # iterative quantum layers
    for v in var:
        qnn_layer(v)

    return qml.expval(qml.X(0))

def get_model(num_modes, num_layers):
    weights = init_weights(num_layers, num_modes)
    shape_tup = weights.shape
    weight_shapes = {'var': shape_tup}
    qlayer = qml.qnn.TorchLayer(quantum_nn, weight_shapes)
    model = torch.nn.Sequential(qlayer)
    return model

num_layers = 2

model = get_model(num_modes, num_layers)
