import numpy as np
import pandas as pd
import torch
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the src directory to the Python path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)


from models.quantum_neural_network import QuantumNeuralNetworkModel
from layers.quantum_data_encoder import QuantumDataEncoder
from layers.weight_initializer import WeightInitializer
from layers.qnn_circuit import qnn_circuit


def load_and_preprocess_data(file_path):
    """
    Load 'financial.csv' and preprocess the data.
    Input: file path
    Output: X_train, X_test, y_train, y_test
    """
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


X_train, X_test, y_train, y_test = load_and_preprocess_data('financial.csv')

num_wires = 8
num_basis = 2
num_layers = 2

encoder = QuantumDataEncoder(num_wires)
encoder.encode(X_train)
weights = WeightInitializer().init_weights(num_layers, num_wires)

shape_tup = weights.shape
weight_shapes = {'var': shape_tup}

model = QuantumNeuralNetworkModel(num_layers,num_wires, qnn_circuit)

# check train_model and evaluate_model in utils.

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

train_model(model, X_train, y_train, batch_size=5, epochs=3)

"""## **5. Evaluation**"""

def evaluate_model(model, X_test, y_test):
    y_pred = model(X_test).detach().numpy()
    y_test = y_test.numpy()
    correct = [1 if p == p_true else 0 for p, p_true in zip(y_pred, y_test)]
    accuracy = sum(correct) / len(y_test)
    print(f"Accuracy: {accuracy * 100}%")

evaluate_model(model, X_test, y_test)
