#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qnn.py

Simple hybrid quantum neural network developed in pennylane(quantum) and pytorch(classical).
"""

import numpy as np
import pennylane as qml

def data_encoding(x):
    """

    """
    num_features = len(x)
    num_wires = 8

    # Squeezing gates
    for i in range(0, min(num_features, num_wires * 2), 2):
        qml.Squeezing(x[i], x[i + 1], wires=i // 2)

    # Beamsplitter gates
    for i in range(num_wires - 1):
        idx = num_wires * 2 + i * 2
        if idx + 1 < num_features:
            qml.Beamsplitter(x[idx], x[idx + 1], wires=[i % num_wires, (i + 1) % num_wires])

    # Rotation gates
    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + i
        if idx < num_features:
            qml.Rotation(x[idx], wires=i)

    # Displacement gates
    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + i * 2
        if idx + 1 < num_features:
            qml.Displacement(x[idx], x[idx + 1], wires=i)

    # Kerr gates
    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + i
        if idx < num_features:
            qml.Kerr(x[idx], wires=i)

    # Squeezing gates (second set)
    for i in range(0, min(num_features - (num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + num_wires), num_wires * 2), 2):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + num_wires + i
        if idx + 1 < num_features:
            qml.Squeezing(x[idx], x[idx + 1], wires=i // 2)

    # Rotation gates (second set)
    for i in range(num_wires):
        idx = num_wires * 2 + (num_wires - 1) * 2 + num_wires + num_wires * 2 + num_wires + num_wires * 2 + i
        if idx < num_features:
            qml.Rotation(x[idx], wires=i)
