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

# examples/hybrid_mnist_classifier.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import pennylane as qml
import time
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

### PREPROCESSING ###

# Define a transform to convert PIL images to tensors and normalize the pixel values
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convert images to tensors with pixel values in range [0, 1]
])

# Download and load the training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)  

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
    depth = 2**4  # 10 classes + 6 zeros for padding
    indices = labels.astype(np.int32)
    one_hot_labels = np.eye(depth)[indices].astype(np.float32)
    return one_hot_labels

# one-hot encoded labels, each label of length cutoff dimension**2
y_train, y_test = one_hot(Y_train), one_hot(Y_test)

# using only 600 samples for training in this experiment
n_samples = 600
test_samples = 100
X_train, X_test, y_train, y_test = X_train[:n_samples], X_test[:test_samples], y_train[:n_samples], y_test[:test_samples]

config.num_wires = 4
config.num_basis = 2
config.probabilities = True
config.multi_output = False
config.single_output = False

# For training
learning_rate = 0.01  # Learning rate for the optimizer
batch_size = 2  # Batch size for DataLoader
device = 'cpu'  # Device to use for training ('cpu' or 'cuda')
num_epochs = 3 
num_layers = 4

# Instantiate classical Model
model = nn.Sequential(
    nn.Flatten(),  # Flatten the input
    nn.Linear(28 * 28, 392),  # Dense layer with 392 units
    nn.ELU(),  # ELU activation function
    nn.Linear(392, 196),  # Dense layer with 196 units
    nn.ELU(),  # ELU activation function
    nn.Linear(196, 98),  # Dense layer with 98 units
    nn.Linear(98, 49),  # Dense layer with 49 units
    nn.ELU(),  # ELU activation function
    nn.Linear(49, 30)  # Dense layer with 30 units
)

# shape weights: 4 layers and 32 parameters per layer
weight_shape = {'var': (4, 32)}

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
print("Total time: {duration:.6f} seconds")

# Evaluate the model
evaluate_model(model, X_test, y_test)
