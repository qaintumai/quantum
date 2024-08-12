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

class ClassicalQuantumClassifier:
    def __init__(self, n_qumodes=4, n_classes=10, activation_function=nn.ELU, dataset='./data', num_layers=4, 
                 learning_rate=0.01, num_epochs=3, batch_size=2, num_samples = 10):
        self.n_qumodes = n_qumodes
        self.n_classes = n_classes
        self.activation_function = activation_function
        self.dataset = dataset
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        # Configuration settings
        self.n_basis = math.ceil(n_classes ** (1 / n_qumodes))
        self.classical_output = 3 * (n_qumodes * 2) + 2 * (n_qumodes - 1)
        self.parameter_count = 5 * n_qumodes + 4 * (n_qumodes - 1)
        config.num_wires = n_qumodes
        config.num_basis = self.n_basis
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
        weight_shape = {'var': (self.num_layers, self.parameter_count)}
        
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
        train_subset = Subset(trainset, range(self.num_samples))
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
        train_model(self.model, criterion, optimizer, train_loader, num_epochs=self.num_epochs, device=self.device)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Total training time: {duration:.6f} seconds")
    
    def evaluate(self):
        _, test_loader = self.load_data()
        X_test, Y_test = next(iter(test_loader))
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)
        
        evaluate_model(self.model, X_test, Y_test)



#example with default values
quantum_model = ClassicalQuantumClassifier()

# Train the model
quantum_model.train()

# Evaluate the model
quantum_model.evaluate()
