#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnn_utils.py

Utility functions for Quantum Neural Networks (QNN) using Pennylane and PyTorch.
"""

import numpy as np
import torch
import pennylane as qml

def prepare_data(data, num_features):
    """
    Prepares and normalizes data for encoding in a quantum neural network.

    Parameters:
    data (array-like): Input data to be prepared.
    num_features (int): Number of features to be encoded.

    Returns:
    array-like: Normalized data suitable for quantum encoding.
    """
    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Normalize data
    max_val = np.max(np.abs(data), axis=1, keepdims=True)
    data = data / max_val
    
    # Pad or truncate data to fit the required number of features
    if data.shape[1] < num_features:
        padding = np.zeros((data.shape[0], num_features - data.shape[1]))
        data = np.hstack((data, padding))
    else:
        data = data[:, :num_features]

    return data

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the performance of a quantum neural network on test data.

    Parameters:
    model: The quantum neural network model to be evaluated.
    test_loader: DataLoader for the test dataset.
    criterion: Loss function.
    device: Device to perform the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
    float: Average loss over the test dataset.
    float: Accuracy of the model on the test dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return test_loss, accuracy

__all__ = [
    'prepare_data',
    'evaluate_model',
]

