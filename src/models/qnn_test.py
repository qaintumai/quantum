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

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from layers.qnn_circuit import qnn_circuit
from utils.utils import train_model, evaluate_model
from layers.weight_initializer import WeightInitializer

class QuantumNeuralNetworkModel:
    def __init__(self, df, num_layers=2, num_wires=8, quantum_nn=qnn_circuit, 
                 learning_rate=0.01, epochs=3, batch_size=2, samples=120, 
                 threshold=0.55, binary=True):
        self.df = df
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.quantum_nn = quantum_nn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = samples
        self.threshold = threshold
        self.binary = binary
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()

    def _build_model(self):
        weights = WeightInitializer.init_weights(self.num_layers, self.num_wires)
        shape_tup = weights.shape
        weight_shapes = {'var': shape_tup}
        qlayer = qml.qnn.TorchLayer(self.quantum_nn, weight_shapes)
        model = nn.Sequential(qlayer)
        model.to(self.device)
        return model

    def preprocess_data(self):
        """
        Preprocess the dataframe, including optional binarization.
        """
        if self.binary:
            self.df.iloc[:, 0][self.df.iloc[:, 0] > self.threshold] = 1.0
            self.df.iloc[:, 0][self.df.iloc[:, 0] <= self.threshold] = 0.0
        
        y = self.df.iloc[:, 0].to_numpy()
        X = self.df.iloc[:, 1:].to_numpy()

        scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = scaler.fit_transform(y.reshape(-1, 1))

        split_index = int(self.samples * 0.8)
        X_train = X[:split_index]
        X_test = X[split_index:self.samples]
        y_train = y_scaled[:split_index]
        y_test = y_scaled[split_index:self.samples]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train = torch.tensor(X_train_scaled, requires_grad=True).float()
        X_test = torch.tensor(X_test_scaled, requires_grad=False).float()
        y_train = torch.tensor(np.reshape(y_train, y_train.shape[0]), requires_grad=False).float()
        y_test = torch.tensor(np.reshape(y_test, y_test.shape[0]), requires_grad=False).float()

        return X_train, X_test, y_train, y_test

    def train(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        start_time = time.time()
        train_model(self.model, criterion, optimizer, train_loader, num_epochs=self.epochs, device=self.device)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Total training time: {duration:.6f} seconds")

    def evaluate(self):
        _, X_test, _, y_test = self.preprocess_data()
        evaluate_model(self.model, X_test, y_test)

# Example usage with the financial dataset
financial_csv_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'financial.csv'))
df = pd.read_csv(financial_csv_path)
df = df.drop(['Company', 'Time'], axis=1)  # Specific to the financial dataset

qnn_model = QuantumNeuralNetworkModel(df=df)
qnn_model.train()
qnn_model.evaluate()
