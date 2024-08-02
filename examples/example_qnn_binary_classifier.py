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

"""
Quantum Neural Network Binary Classifier Example

This script demonstrates the training and evaluation of a Quantum Neural Network (QNN) for binary classification on financial distress data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the src directory to the Python path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# from models.quantum_neural_network import QuantumNeuralNetworkModel  /Self-referential
# from layers.quantum_data_encoder import QuantumDataEncoder  /This is inside qnn_circuit
from layers.weight_initializer import WeightInitializer
from layers.qnn_circuit import qnn_circuit
from utils.utils import train_model, evaluate_model

def load_and_preprocess_data(file_path):
    """
    Load 'financial.csv' and preprocess the data.
    Input: file path
    Output: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(file_path)
    df = df.drop(['Company', 'Time'], axis=1)
    df.iloc[:, 0][df.iloc[:, 0] > 0.55] = 1.0
    df.iloc[:, 0][df.iloc[:, 0] <= 0.55] = 0.0

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    X = X.to_numpy()
    y = y.to_numpy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    X_train = X[:100]
    X_test = X[100:120]
    y_train = y_scaled[:100]
    y_test = y_scaled[100:120]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = torch.tensor(X_train_scaled, requires_grad=True).float()
    X_test = torch.tensor(X_test_scaled, requires_grad=False).float()
    y_train = torch.tensor(np.reshape(y_train, y_train.shape[0]), requires_grad=False).float()
    y_test = torch.tensor(np.reshape(y_test, y_test.shape[0]), requires_grad=False).float()

    return X_train, X_test, y_train, y_test

# Find the right path for 'financial.csv'
financial_csv_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'financial.csv'))
X_train, X_test, y_train, y_test = load_and_preprocess_data(financial_csv_path)

# Creating DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Configuration
# For the quantum neural network circuit
num_wires = 8
num_basis = 2
num_layers = 2

# For training
learning_rate = 0.01  # Learning rate for the optimizer
batch_size = 2  # Batch size for DataLoader
device = 'cpu'  # Device to use for training ('cpu' or 'cuda')

# Instantiate the QNN model
quantum_nn_model = QuantumNeuralNetworkModel(num_layers, num_wires, qnn_circuit)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(quantum_nn_model.parameters(), lr=learning_rate)

# Train the model
train_model(quantum_nn_model, criterion, optimizer, train_loader, num_epochs=num_epochs, device=device)

# Evaluate the model
evaluate_model(quantum_nn_model, X_test, y_test)
