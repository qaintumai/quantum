import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Training")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    return args

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = Config(json.load(f))

    # Load training logs
    log_file = config.general.log_dir + "train_qnn.log"
    log_data = pd.read_csv(log_file)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(log_data['epoch'], log_data['train_loss'], label='Training Loss')
    plt.plot(log_data['epoch'], log_data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
