import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
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
from layers.qnn_layer import QuantumNeuralNetworkLayer
from utils import config

### CONFIGURATION ###
n_qumodes = 4  # Set the number of qumodes here
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

# Define a transform to convert PIL images to tensors and normalize the pixel values
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convert images to tensors with pixel values in range [0, 1]
])

# Download and load the training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Use only the first 600 samples for training
n_samples = 2
train_subset = Subset(trainset, range(n_samples))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=2, shuffle=True)

# Download and load the test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True)

# Extract the full dataset from the DataLoader (this loads the entire dataset into memory)
X_train, Y_train = next(iter(train_loader))
X_test, Y_test = next(iter(test_loader))

# Convert images to numpy arrays and ensure they are of type float
X_train = X_train.numpy().astype(np.float32)
X_test = X_test.numpy().astype(np.float32)

# Convert labels to numpy arrays and ensure they are of type float
Y_train = Y_train.numpy().astype(np.float32)
Y_test = Y_test.numpy().astype(np.float32)

# Print shapes to verify
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Verify that pixel values are in the range [0, 1]
print("X_train min and max values:", X_train.min(), X_train.max())
print("X_test min and max values:", X_test.min(), X_test.max())

# One hot encoding, necessary for file
def one_hot(labels):  
    depth = n_basis**n_qumodes  # Adjust depth based on the number of qumodes
    indices = labels.astype(np.int32)
    one_hot_labels = np.eye(depth)[indices].astype(np.float32)
    return one_hot_labels

# one-hot encoded labels, each label of length cutoff dimension**2
y_train, y_test = one_hot(Y_train), one_hot(Y_test)


# For training
learning_rate = 0.01  # Learning rate for the optimizer
batch_size = 2  # Batch size for DataLoader
device = 'cpu'  # Device to use for training ('cpu' or 'cuda')
num_epochs = 1 
num_layers = 4

# Instantiate classical Model
#This is an example of a classical model, the user can define internal layering and activation functions (classical parameters: num layers, activation function, input size)
model = nn.Sequential(
    nn.Flatten(),  # Flatten the input
    nn.Linear(28 * 28, 392),  # Dense layer with 392 units
    nn.ELU(),  # ELU activation function
    nn.Linear(392, 196),  # Dense layer with 196 units
    nn.ELU(),  # ELU activation function
    nn.Linear(196, 98),  # Dense layer with 98 units
    nn.ELU(),  # ELU activation function
    nn.Linear(98, classical_output),  # Dense layer
)

# shape weights: adjust based on number of layers and qumodes
weight_shape = {'var': (num_layers, parameter_count)}

# Define the quantum layer using TorchLayer
quantum_layer = qml.qnn.TorchLayer(qnn_circuit, weight_shape)
# add to the classical sequential model
model.add_module('quantum_layer', quantum_layer)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
start_time = time.time()
train_model(model, criterion, optimizer, train_loader, num_epochs=num_epochs, device=device)
end_time = time.time()
duration = end_time - start_time
print(f"Total time: {duration:.6f} seconds")

# Evaluate the model
evaluate_model(model, X_test, y_test)
