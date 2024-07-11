from .qnn_circuit import QuantumNeuralNetwork

def main():
    qnn = QuantumNeuralNetwork(num_wires=8)
    x = np.random.rand(16)
    qnn.data_encoding(x)
    print("Data encoding completed.")

if __name__ == "__main__":
    main()
