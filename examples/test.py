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

# Add the src directory to the Python path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from layers.quantum_data_encoder import QuantumDataEncoder
from layers.qnn_circuit import qnn_circuit
from utils.utils import train_model, evaluate_model
from layers.quantum_layer import QuantumNeuralNetworkLayer
from utils import config

### CONFIGURATION ###
n_qumodes = 4
num_classes = 10
n_basis = math.ceil(num_classes ** (1 / n_qumodes))
classical_output = 3 * (n_qumodes * 2) + 2 * (n_qumodes - 1)
parameter_count = 5 * n_qumodes + 4 * (n_qumodes - 1)
config.num_wires = n_qumodes
config.num_basis = n_basis
config.probabilities = True
config.multi_output = False
config.single_output = False

### PREPROCESSING ###

# Transform for MNIST dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Load and subset the training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
n_samples = 10
train_subset = Subset(trainset, range(n_samples))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=2, shuffle=True)

# Print the data type of train_loader
print(f"Data type of train_loader: {type(train_loader)}")

# Load the test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True)

# For training
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 1
num_layers = 4

# Instantiate the model
model = nn.Sequential(
    nn.Flatten(),  # Flatten the input
    nn.Linear(28 * 28, 392),  # Dense layer with 392 units
    nn.ELU(),  # ELU activation function
    nn.Linear(392, 196),  # Dense layer with 196 units
    nn.ELU(),  # ELU activation function
    nn.Linear(196, 98),  # Dense layer with 98 units
    nn.ELU(),  # ELU activation function
    nn.Linear(98, classical_output),  # Dense layer with classical_output units
)

# Adjust weights based on number of layers and qumodes
weight_shape = {'var': (num_layers, parameter_count)}

# Define the quantum layer using TorchLayer
quantum_layer = qml.qnn.TorchLayer(qnn_circuit, weight_shape)
model.add_module('quantum_layer', quantum_layer)

# Move model to the appropriate device
model.to(device)

# Get the full test set from the test_loader
X_test, Y_test = next(iter(test_loader))
X_test, Y_test = X_test.to(device), Y_test.to(device)

# Print the data types of X_test and Y_test
print(f"Data type of X_test: {type(X_test)}, Data type of Y_test: {type(Y_test)}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
start_time = time.time()
train_model(model, criterion, optimizer, train_loader, num_epochs=num_epochs, device=device)
end_time = time.time()
duration = start_time - end_time
print(f"Total time: {duration:.6f} seconds")

# Evaluate the model
evaluate_model(model, X_test, Y_test)
