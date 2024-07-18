class QuantumNeuralNetworkModel:
    def __init__(self, num_layers, num_wires):
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.model = self._build_model()

    def _build_model(self):
        weights = WeightInitializer.init_weights(self.num_layers, self.num_wires)
        shape_tup = weights.shape
        weight_shapes = {'var': shape_tup}
        qlayer = qml.qnn.TorchLayer(quantum_nn, weight_shapes)
        model = Sequential(qlayer)
        return model

# Example usage
num_layers = 2
num_wires = 6
qnn_model = QuantumNeuralNetworkModel(num_layers, num_wires)
model = qnn_model.model
