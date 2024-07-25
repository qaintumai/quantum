import argparse
import json
import torch
from torch.utils.data import DataLoader
from src.utils.data_loader import load_data
from src.utils.config import Config
from src.models.quantum_neural_network import QuantumNeuralNetwork
from src.models.quantum_transformer import QuantumTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Predict using Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output', type=str, required=True, help='Path to save predictions')
    args = parser.parse_args()
    return args

def main(config_path, input_path, output_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = Config(json.load(f))

    # Load data
    data = load_data(input_path)
    data_loader = DataLoader(data, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)

    # Initialize model
    if config.model.type == 'qnn':
        model = QuantumNeuralNetwork(input_dim=config.qnn.input_dim,
                                     hidden_dim=config.qnn.hidden_dim,
                                     output_dim=config.qnn.output_dim,
                                     num_layers=config.qnn.num_layers,
                                     activation=config.qnn.activation,
                                     dropout=config.qnn.dropout)
    elif config.model.type == 'qt':
        model = QuantumTransformer(input_dim=config.qt.input_dim,
                                   num_heads=config.qt.num_heads,
                                   num_encoder_layers=config.qt.num_encoder_layers,
                                   num_decoder_layers=config.qt.num_decoder_layers,
                                   dim_feedforward=config.qt.dim_feedforward,
                                   dropout=config.qt.dropout)

    # Load model weights
    model.load_state_dict(torch.load(config.model.load_model_path))
    model.eval()

    # Make predictions
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    # Save predictions
    with open(output_path, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.input, args.output)
