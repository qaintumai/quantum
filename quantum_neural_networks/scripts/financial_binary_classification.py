#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary classification of financial distress in financial.csv
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
import torch
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import the functions from the module
from models import data_encoding, qnn_layer, init_weights, get_model

def load_and_preprocess_data(file_path):
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

# Ensure the correct path to the data file
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'financial.csv')
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

num_layers = 2
num_wires = 8  # This should match num_wires in the imported module

model = get_model(num_wires, num_layers)

def train_model(model, X_train, y_train, batch_size=5, epochs=6):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True, drop_last=True
    )

    for epoch in range(epochs):
        running_loss = 0
        for xs, ys in data_loader:
            optimizer.zero_grad()
            x = model(xs).float()
            y = ys.float()
            loss = loss_fn(x, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(data_loader)
        print(f"Average loss over epoch {epoch + 1}: {avg_loss:.4f}")

def evaluate_model(model, X_test, y_test):
    y_pred = model(X_test).detach().numpy()
    y_test = y_test.numpy()
    correct = [1 if round(p) == p_true else 0 for p, p_true in zip(y_pred, y_test)]
    accuracy = sum(correct) / len(y_test)
    print(f"Accuracy: {accuracy * 100}%")

train_model(model, X_train, y_train, batch_size=5, epochs=6)
evaluate_model(model, X_test, y_test)
