#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Script to train and evaluate the quantum neural network model.
"""

import torch
import numpy as np
import pennylane as qml
from models.qnn_layer import data_encoding, qnn_layer, init_weights
from models.qnn_model import get_model
from models.qnn_utils import preprocess_data, train_model, evaluate_model, visualize_training

# Parameters
num_wires = 8
num_basis = 2
num_layers = 2
batch_size = 5
epochs = 6

# Assume X_train and y_train are predefined training data
# For the purpose of this example, let's create dummy data
X_train = np.random.rand(100, num_wires)
y_train = np.random.rand(100, 1)
X_test = np.random.rand(20, num_wires)
y_test = np.random.rand(20, 1)

# Preprocess the data
X_train_preprocessed = np.array([preprocess_data(x, num_wires) for x in X_train])
X_test_preprocessed = np.array([preprocess_data(x, num_wires) for x in X_test])

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_preprocessed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_preprocessed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training
train_data_loader = torch.utils.data.DataLoader(
    list(zip(X_train_tensor, y_train_tensor)), batch_size=batch_size, shuffle=True, drop_last=True
)

# Create the model
model = get_model(num_wires, num_layers)

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
print("Training the model...")
loss_history = train_model(model, train_data_loader, loss_fn, optimizer, epochs)

# Visualize the training loss
visualize_training(loss_history)

# Evaluate the model on the test set
test_data_loader = torch.utils.data.DataLoader(
    list(zip(X_test_tensor, y_test_tensor)), batch_size=batch_size, shuffle=False, drop_last=True
)
test_loss = evaluate_model(model, test_data_loader, loss_fn)
print(f"Test Loss: {test_loss:.4f}")

# Example prediction
example_input = torch.tensor(np.random.rand(1, num_wires), dtype=torch.float32)
output = model(example_input)
print("Example prediction:", output)
