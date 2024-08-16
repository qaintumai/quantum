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

from layers.qnn_circuit import qnn_circuit
# from utils.utils import train_model, evaluate_model
from utils import config


def train_model(model, criterion, optimizer, train_loader, num_epochs=100, device='cpu', debug=True):

    model.to(device)  # Move model to the specified device
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        operation_count = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            print("____LABELS______")
            print(labels)

            print("_____INPUTS_____")
            print(inputs)

            # Forward pass
            outputs = model(inputs)

            print("____OUTPUTS______")
            print(outputs)

            loss = criterion(outputs, labels)


            # Backward pass and optimization
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            if debug:
                print("Performed backpropagation and optimization step")

            running_loss += loss.item() * inputs.size(0)
            operation_count += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or debug:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        if debug:
            print(f"Total operations in epoch {epoch + 1}: {operation_count}")

    print("Training complete")


def evaluate_model(model, X_test, y_test):

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_pred = model(X_test).detach().numpy()
        print(y_pred)
        y_test = y_test.numpy()
        correct = [1 if p == p_true else 0 for p, p_true in zip(y_pred, y_test)]
        accuracy = sum(correct) / len(y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")



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
    nn.ELU(),  # ELU activation function
    nn.Linear(98, 49),  # Dense layer with 49 units
    nn.ELU(),  # ELU activation function
    nn.Linear(49, 30)  # Dense layer with 30 units
)

# shape weights: 4 layers and 32 parameters per layer
weight_shape = {'var': (4, 32)}

# Define the quantum layer using TorchLayer
quantum_layer = qml.qnn.TorchLayer(qnn_circuit, weight_shape)
# add to the classical sequential model
model.add_module('8', quantum_layer)

print(model)
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
