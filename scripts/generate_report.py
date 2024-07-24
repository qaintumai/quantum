import argparse
import json
import pandas as pd
from src.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Model Report")
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

    # Generate report
    report = {
        "project_name": config.general.project_name,
        "num_epochs": config.training.num_epochs,
        "final_train_loss": log_data['train_loss'].iloc[-1],
        "final_val_loss": log_data['val_loss'].iloc[-1],
        "best_val_loss": log_data['val_loss'].min(),
    }

    report_file = config.general.log_dir + "report.json"
    with open(report_file, 'w') as f:
        json.dump(report
