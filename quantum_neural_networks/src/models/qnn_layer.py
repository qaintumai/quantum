#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnn.py

quantum neural network layer developed in Pennylane.
"""

import numpy as np
import pennylane as qml

def data_encoding(x):
    num_wires=8
    num_features = len(x)

    for i in range(0, min(num_features, num_wires * 2), 2):
        qml.Squeezing(x[i], x[i + 1], wires=i // 2)

    for i in range(num_wires - 1):
        idx = num_wires * 2 + i * 2
        if idx + 1 < num_features:
            qml.Beamsplitter(x[idx], x[idx + 1], wires=[i % num_wires, (i + 1) % num_wires])

    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + i
        if idx < num_features:
            qml.Rotation(x[idx], wires=i)

    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + i * 2
        if idx + 1 < num_features:
            qml.Displacement(x[idx], x[idx + 1], wires=i)

    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + i
        if idx < num_features:
            qml.Kerr(x[idx], wires=i)

    for i in range(0, min(num_features - (num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + num_wires), num_wires * 2), 2):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + num_wires + i
        if idx + 1 < num_features:
            qml.Squeezing(x[idx], x[idx + 1], wires=i // 2)

    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + num_wires + num_wires * 2 + i
        if idx < num_features:
            qml.Rotation(x[idx], wires=i)

def qnn_layer(v):
    num_wires=8
    num_params = len(v)

    for i in range(num_wires - 1):
        idx = i * 2
        if idx + 1 < num_params:
            theta = v[idx]
            phi = v[idx + 1]
            qml.Beamsplitter(theta, phi, wires=[i % num_wires, (i + 1) % num_wires])

    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + i
        if idx < num_params:
            qml.Rotation(v[idx], wires=i)

    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + i
        if idx < num_params:
            qml.Squeezing(v[idx], 0.0, wires=i)

    for i in range(num_wires - 1):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + i * 2
        if idx + 1 < num_params:
            theta = v[idx]
            phi = v[idx + 1]
            qml.Beamsplitter(theta, phi, wires=[i % num_wires, (i + 1) % num_wires])

    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + (num_wires - 1) * 2 + i
        if idx < num_params:
            qml.Rotation(v[idx], wires=i)

    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + (num_wires - 1) * 2 + num_wires + i
        if idx < num_params:
            qml.Displacement(v[idx], 0.0, wires=i)

    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + (num_wires - 1) * 2 + num_wires + num_wires + i
        if idx < num_params:
            qml.Kerr(v[idx], wires=i)

def init_weights(layers, num_wires, active_sd=0.0001, passive_sd=0.1):
    M = (modes-1)*2 + num_wires

    int1_weights = np.random.normal(size=[layers, M], scale=passive_sd)
    s_weights = np.random.normal(size=[layers, num_wires], scale=active_sd)
    int2_weights = np.random.normal(size=[layers, M], scale=passive_sd)
    dr_weights = np.random.normal(size=[layers, num_wires], scale=active_sd)
    k_weights = np.random.normal(size=[layers, num_wires], scale=active_sd)

    weights = np.concatenate([int1_weights, s_weights, int2_weights, dr_weights, k_weights], axis=1)

    return weights
