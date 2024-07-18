import pennylane as qml

"""
This module defines the QuantumDataEncoder class, which is used to encode classical data 
into a quantum format suitable for use in a quantum neural network. The class applies a 
series of quantum gates (squeezing, beamsplitter, rotation, displacement, and Kerr gates) 
to the input data.

Usage:
To use the QuantumDataEncoder class, import it as follows:
    from layers.quantum_data_encoding import QuantumDataEncoder

Example:
    encoder = QuantumDataEncoder(num_wires=8)
    encoder.encode(input_data)
"""

class QuantumDataEncoder:
    """
    A class used to encode classical data into a quantum format using various quantum gates.
    """

    def __init__(self, num_wires):
        """
        Initializes the QuantumDataEncoder class with the given number of wires.

        Parameters:
        - num_wires (int): Number of quantum wires. 

        NOTE: currently, we only support num_wires =8 or 6, which is declared in the qnn classes.
        """
        self.num_wires = num_wires

    def encode(self, x):
        num_features = len(x)

        # Squeezing gates
        for i in range(0, min(num_features, self.num_wires * 2), 2):
            qml.Squeezing(x[i], x[i + 1], wires=i // 2)

        # Beamsplitter gates
        for i in range(self.num_wires - 1):
            idx = self.num_wires * 2 + i * 2
            if idx + 1 < num_features:
                qml.Beamsplitter(x[idx], x[idx + 1], wires=[i % self.num_wires, (i + 1) % self.num_wires])

        # Rotation gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + i
            if idx < num_features:
                qml.Rotation(x[idx], wires=i)

        # Displacement gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + i * 2
            if idx + 1 < num_features:
                qml.Displacement(x[idx], x[idx + 1], wires=i)

        # Kerr gates
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + i
            if idx < num_features:
                qml.Kerr(x[idx], wires=i)

        # Squeezing gates (second set)
        for i in range(0, min(num_features - (self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires), self.num_wires * 2), 2):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires + i
            if idx + 1 < num_features:
                qml.Squeezing(x[idx], x[idx + 1], wires=i // 2)

        # Rotation gates (second set)
        for i in range(self.num_wires):
            idx = self.num_wires * 2 + (self.num_wires - 1) * 2 + self.num_wires + self.num_wires * 2 + self.num_wires + self.num_wires * 2 + i
            if idx < num_features:
                qml.Rotation(x[idx], wires=i)
