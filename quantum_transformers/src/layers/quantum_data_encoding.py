"""
Quantum Data Encoding Layer for Quantum Feedforward using Quantum Neural Networks (QNN).
"""

from ..src/models import data_encoding

class QuantumDataEncoding:
    def __init__(self, num_wires=8):
        self.qnn = QuantumNeuralNetwork(num_wires=num_wires)

    def encode(self, x):
        """
        Encodes the input data using the QuantumNeuralNetwork's data_encoding method.

        Parameters:
        x (array-like): Input data to be encoded.
        """
        self.qnn.data_encoding(x)
        # Additional functionality or return value can be added here
