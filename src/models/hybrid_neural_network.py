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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import math
import pennylane as qml
import time
from torch.utils.data import Subset

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
if src_dir not in sys.path:
    sys.path.append(src_dir)
from layers.qnn_circuit import qnn_circuit
from utils.utils import train_model, evaluate_model
from utils import config

class HybridNeuralNetwork:
    def __init__(self, qumodes=4, classes=10, activation_function=nn.ELU, dataset='./data', q_layers=4, 
                 learning_rate=0.01, epochs=3, batch_size=2, samples = 10):
        self.qumodes = qumodes
        self.classes = classes
        self.activation_function = activation_function
        self.dataset = dataset
        self.q_layers = q_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = samples
        
        # Configuration settings
        self.basis = math.ceil(classes ** (1 / qumodes))
        self.classical_output = 3 * (qumodes * 2) + 2 * (qumodes - 1)
        self.parameter_count = 5 * qumodes + 4 * (qumodes - 1)
        config.num_wires = qumodes
        config.num_basis = self.basis
        config.probabilities = True
        config.multi_output = False
        config.single_output = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the model
        self.model = self.build_model()

    def build_model(self):
        model = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(28 * 28, 392),  # Dense layer with 392 units
            self.activation_function(),  # Activation function
            nn.Linear(392, 196),  # Dense layer with 196 units
            self.activation_function(),  # Activation function
            nn.Linear(196, 98),  # Dense layer with 98 units
            self.activation_function(),  # Activation function
            nn.Linear(98, self.classical_output),  # Dense layer with classical_output units
        )
        
        # Adjust weights based on number of layers and qumodes
        weight_shape = {'var': (self.q_layers, self.parameter_count)}
        
        # Define the quantum layer using TorchLayer
        quantum_layer = qml.qnn.TorchLayer(qnn_circuit, weight_shape)
        model.add_module('quantum_layer', quantum_layer)
        
        # Move model to the appropriate device
        model.to(self.device)
        
        return model

    def load_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        # Load and subset the training dataset
        trainset = torchvision.datasets.MNIST(root=self.dataset, train=True, download=True, transform=transform)
        train_subset = Subset(trainset, range(self.samples))
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)

        # Load the test dataset
        testset = torchvision.datasets.MNIST(root=self.dataset, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        
        return train_loader, test_loader

    def train(self):
        train_loader, test_loader = self.load_data()
        
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
        _, test_loader = self.load_data()
        X_test, Y_test = next(iter(test_loader))
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)
        
        evaluate_model(self.model, X_test, Y_test)



#example with default values
quantum_model = HybridNeuralNetwork()

# Train the model
quantum_model.train()

# Evaluate the model
quantum_model.evaluate()
