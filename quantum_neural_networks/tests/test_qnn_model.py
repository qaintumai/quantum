import unittest
import numpy as np
from quantum_neural_networks.src.models.qnn_model import QuantumNeuralNetwork

class TestQNNModel(unittest.TestCase):
    def setUp(self):
        self.qnn = QuantumNeuralNetwork(num_wires=4)

    def test_data_encoding(self):
        x = np.random.randn(8)
        self.qnn.data_encoding(x)
        # Add assertions to verify data encoding

    def test_qnn_layer(self):
        v = np.random.randn(20)
        self.qnn.qnn_layer(v)
        # Add assertions to verify QNN layer

    def test_init_weights(self):
        layers = 2
        modes = 4
        weights = self.qnn.init_weights(layers, modes)
        self.assertEqual(weights.shape, (layers, (modes - 1) * 2 + modes + modes + (modes - 1) * 2 + modes + modes))

if __name__ == '__main__':
    unittest.main()

