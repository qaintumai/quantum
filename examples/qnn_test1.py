import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import pennylane as qml
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.quantum_neural_network import QuantumNeuralNetwork  
from layers.qnn_circuit import qnn_circuit
from utils.utils import train_model, evaluate_model
from utils.config import set_config_variable, num_layers, num_wires, single_output
"""

Dataset information
Input Variables (x):

    Number of times pregnant
    Plasma glucose concentration at 2 hours in an oral glucose tolerance test
    Diastolic blood pressure (mm Hg)
    Triceps skin fold thickness (mm)
    2-hour serum insulin (μIU/ml)
    Body mass index (weight in kg/(height in m)2)
    Diabetes pedigree function
    Age (years)

Output Variables (y):

    Class label (0 or 1)

"""

#LOADING DATA
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data/pima-indians-diabetes.csv')
dataset = np.loadtxt(data_path, delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

print(f"Size of X tensor: {X.size()} and first element of X: {X[0]}")
print(f"Size of Y tensor: {y.size()} and first element of Y: {y[0]}")



#CREATING MODEL
set_config_variable('single_output', True)
set_config_variable('num_layers', 2)
set_config_variable('num_wires', 8)
set_config_variable('num_basis', 2)

print(f"TESTING set config var: (should be 2) {num_layers}")
print(f"TESTING set config var: (should be True) {single_output}")

quantum_nn = QuantumNeuralNetwork(num_layers, num_wires, qnn_circuit).qlayers
qnn_model = torch.nn.Sequential(quantum_nn)

print(f"Looking at the model: {qnn_model}")


#TRAINING MODEL
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(qnn_model.parameters(), lr=0.01)

n_epochs = 3
batch_size = 10
 
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        print(f"Batch number: {i} in Epoch {epoch}")
        Xbatch = X[i:i+batch_size]
        print(f"Size of Xbatch: {Xbatch.size()}")
        #ISSUE > > > Mismatching shapes
        y_pred = qnn_model(Xbatch).reshape(-1, 1)  # Reshape y_pred to (10, 1)
        print(f"Size of y_pred: {y_pred.size()}, Values: {y_pred}")

        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        print(f"loss = {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Finished epoch {epoch}, latest loss {loss}')

# EVALUATING
with torch.no_grad():
    y_pred = qnn_model(X)

    print(f"Size of y_pred (evaluation): {y_pred.size()}, Values: {y_pred}")

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy: {accuracy}")
