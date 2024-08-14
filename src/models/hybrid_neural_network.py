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
    """
    This class implements a hybrid quantum-classical neural network model using PyTorch for classical layers
    and PennyLane for quantum layers. The class handles the construction of the model, data loading, training,
    and evaluation processes.

    Usage:
    To use the HybridNeuralNetwork class, create an instance of the class and call the train and evaluate methods.

    Example:
    quantum_model = HybridNeuralNetwork(qumodes=4, classes=10)
    quantum_model.train(epochs=2, batch_size=16, samples=200)
    quantum_model.evaluate()
    """

    def __init__(self, qumodes=4, classes=10, activation_function=nn.ELU, q_layers=4):
        """
        Initializes the HybridNeuralNetwork class with the given configuration parameters.

        Parameters:
        - qumodes (int): Number of quantum modes (wires) in the quantum circuit.
        - classes (int): Number of output classes for the classification task.
        - activation_function (torch.nn.Module): Activation function to use in the classical layers. Default is nn.ELU.
        - q_layers (int): Number of layers in the quantum circuit.
        """
        self.qumodes = qumodes
        self.classes = classes
        self.activation_function = activation_function
        self.q_layers = q_layers
        
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
        
        # Training parameters (initialized later in the train method)
        self.dataset = None
        self.learning_rate = None
        self.epochs = None
        self.batch_size = None
        self.samples = None
        
        # Build the model
        self.model = self.build_model()

    def build_model(self):
        """
        Constructs the hybrid neural network model, consisting of classical layers followed by a quantum layer.

        Returns:
        - model (torch.nn.Sequential): The constructed neural network model.
        """
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

    def load_data(self, dataset=None, samples=None, batch_size=None):
        """
        Loads the MNIST dataset and prepares data loaders for training and testing.

        Parameters:
        - dataset (str): Path to the dataset directory. Defaults to self.dataset.
        - samples (int): Number of samples to use for training. Defaults to self.samples.
        - batch_size (int): Batch size for loading data. Defaults to self.batch_size.

        Returns:
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        """
        if dataset is None:
            dataset = self.dataset
        if samples is None:
            samples = self.samples
        if batch_size is None:
            batch_size = self.batch_size
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        # Load and subset the training dataset
        trainset = torchvision.datasets.MNIST(root=dataset, train=True, download=True, transform=transform)
        train_subset = Subset(trainset, range(samples))
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # Load the test dataset
        testset = torchvision.datasets.MNIST(root=dataset, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        
        return train_loader, test_loader

    def train(self, dataset='./data', learning_rate=0.01, epochs=3, batch_size=2, samples=10):
        """
        Trains the hybrid neural network model on the MNIST dataset.

        Parameters:
        - dataset (str): Path to the dataset directory.
        - learning_rate (float): Learning rate for the optimizer.
        - epochs (int): Number of epochs for training.
        - batch_size (int): Batch size for training.
        - samples (int): Number of samples to use for training.
        """
        # Update the instance attributes with the values passed to train
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = samples
        
        train_loader, _ = self.load_data()
        
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
        """
        Evaluates the trained hybrid neural network model on the test dataset.

        The evaluation process uses the parameters set during the training phase (dataset path, batch size, and
        number of samples).
        """
        _, test_loader = self.load_data()
        X_test, Y_test = next(iter(test_loader))
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)
        
        evaluate_model(self.model, X_test, Y_test)


# Example usage with default values
quantum_model = HybridNeuralNetwork()
quantum_model.train(epochs=2, batch_size=16, samples=200)
quantum_model.evaluate()
