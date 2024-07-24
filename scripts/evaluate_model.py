import argparse
import json
import torch
from torch.utils.data import DataLoader
from src.utils.data_loader import load_data
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.quantum_neural_network import QuantumNeuralNetwork
from src.models.quantum_transformer import QuantumTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    return args

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = Config(json.load(f))

    # Set up logger
    logger = setup_logger(config.general.log_dir, "evaluate_model.log")

    # Load data
    logger.info("Loading data...")
    data = load_data(config.data.test_data_path)
    test_loader = DataLoader(data, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)

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

    # Evaluation
    logger.info("Evaluating model...")
    criterion = getattr(nn, config.training.loss_function)()
    total_loss = 0.0
    for inputs, labels in test_loader:
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_loss}")

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
