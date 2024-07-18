import pennylane as qml

# Define the number of wires and basis states
num_wires = 8
num_basis = 2

# Select a device
dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=num_basis)

@qml.qnode(dev, interface="torch")
def qnn_multi_output(inputs, var):
    encoder = QuantumDataEncoder(num_wires)
    encoder.encode(inputs)

    # Iterative quantum layers
    q_layer = QuantumNeuralNetworkLayer(num_wires)
    for v in var:
        q_layer.apply(v)

    # Return the probabilities for all wires
    return [qml.expval(qml.X(wire)) for wire in range(num_wires)]
