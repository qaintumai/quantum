import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.models.quantum_transformer import QuantumTransformer
from src.utils.data_loader import load_data
from src.utils.config import Config
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train Quantum Transformer")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    return args

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = Config(json.load(f))

    # Set up logger
    logger = setup_logger(config.general.log_dir, "train_qt.log")

    # Set random seed for reproducibility
    torch.manual_seed(config.general.seed)

    # Load data
    logger.info("Loading data...")
    data = load_data(config.data.train_data_path)
    train_data, val_data = train_test_split(data, test_size=config.training.validation_split, random_state=config.general.seed)

    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, shuffle=config.data.shuffle, num_workers=config.data.num_workers)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)

    # Initialize model
    model = QuantumTransformer(input_dim=config.qt.input_dim,
                               num_heads=config.qt.num_heads,
                               num_encoder_layers=config.qt.num_encoder_layers,
                               num_decoder_layers=config.qt.num_decoder_layers,
                               dim_feedforward=config.qt.dim_feedforward,
                               dropout=config.qt.dropout)

    # Loss function and optimizer
    criterion = getattr(nn, config.training.loss_function)()
    optimizer = getattr(optim, config.training.optimizer)(model.parameters(), lr=config.training.learning_rate)

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(config.training.num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}, Training Loss: {avg_train_loss}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}, Validation Loss: {avg_val_loss}")

        # Check for improvement
        if config.model.save_best_only and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.model.save_dir, "best_model.pth"))
            logger.info(f"Model saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(config.model.save_dir, "final_model.pth"))
    logger.info("Training completed and model saved.")

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
