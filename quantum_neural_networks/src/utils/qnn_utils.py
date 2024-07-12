#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qnn_utils.py

Utility functions for the quantum neural network (QNN) project.
"""

import numpy as np
import torch

def preprocess_data(data, num_wires):
    """
    Preprocess the input data to fit the quantum neural network requirements.
    
    Parameters:
    data (array-like): The input data to be preprocessed.
    num_wires (int): The number of quantum wires to use in the model.
    
    Returns:
    array: Preprocessed data suitable for input into the QNN.
    """
    # Example preprocessing: normalize the data
    data = np.array(data)
    max_val = np.max(np.abs(data), axis=0)
    data = data / max_val
    # Ensure data fits into the number of wires
    if len(data) > num_wires:
        data = data[:num_wires]
    return data

def train_model(model, data_loader, loss_fn, optimizer, epochs):
    """
    Train the PyTorch model.
    
    Parameters:
    model (torch.nn.Module): The PyTorch model to be trained.
    data_loader (torch.utils.data.DataLoader): The data loader for the training data.
    loss_fn (torch.nn.Module): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    epochs (int): The number of training epochs.
    
    Returns:
    list: Training loss over epochs.
    """
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return loss_history

def evaluate_model(model, data_loader, loss_fn):
    """
    Evaluate the PyTorch model.
    
    Parameters:
    model (torch.nn.Module): The PyTorch model to be evaluated.
    data_loader (torch.utils.data.DataLoader): The data loader for the evaluation data.
    loss_fn (torch.nn.Module): The loss function.
    
    Returns:
    float: Evaluation loss.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def visualize_training(loss_history):
    """
    Visualize the training loss over epochs.
    
    Parameters:
    loss_history (list): The training loss history.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()
