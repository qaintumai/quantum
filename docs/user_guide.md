# User Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Usage](#usage)
    - [Quantum Neural Networks (QNN)](#quantum-neural-networks-qnn)
    - [Quantum Transformers (QT)](#quantum-neural-networks-qnn)
5. [Example Notebooks](#example-notebooks)
    - [Layers](#layers)
    - [Models](#models)
    - [Utilities](#utilities)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## Introduction

Welcome to the Quantum Project! This guide provides an overview of the Quantum Neural Networks (QNN) and Quantum Transformers (QT) project. It includes instructions for setting up the project, an explanation of the project structure, and examples of how to use the various components.

---

## Project Structure

The project is organized into the following main directories:

- **/docs**: Documentation files
- **/src**: Source code
  - **/layers**: Contains various layer implementations
    - `input_embedding.py`
    - `multi_headed_attention.py`
    - `qnn_multi_output.py`
    - `qnn_probabilities.py`
    - `qnn_single_output.py`
    - `quantum_data_encoding.py`
    - `quantum_feed_forward.py`
    - `quantum_layer.py`
    - `scaled_dot_product.py`
    - `weight_initializer.py`
  - **/models**: Contains model definitions
    - `quantum_decoder.py`
    - `quantum_encoder.py`
    - `quantum_neural_network.py`
    - `quantum_transformer.py`
  - **/utils**: Utility functions and scripts
- **/tests**: Test files
- **/scripts**: Scripts for training and evaluating models

## Setup Instructions

### **Prerequisites**

Ensure you have the following installed:

- Python 3.10+
- Required Python packages (listed in `requirements.txt`)


### **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/quantum-project.git
    cd quantum-project
    ```
2. Create and avtivate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Usage

### **Quantum Neural Networks (QNN)**

#### **Training a QNN Model**

To train a QNN model, use the `train_qnn.py` script located in the `/scripts` directory. Here's an example command:

```bash
python scripts/train_qnn.py --config configs/qnn_config.json
```

The configuration file (qnn_config.json) should specify the necessary parameters for training, such as the dataset path, model hyperparameters, and training settings.

#### **Evaluating a QNN Model**

To evaluate a trained QNN model, use the evaluate_qnn.py script:

```bash
python scripts/evaluate_qnn.py --model_path models/qnn_model.pth --config configs/qnn_config.json
```
### **Quantum Transformers**

#### **Training a QT Model**

To train a QT model, use the train_qt.py script:

```bash
python scripts/train_qt.py --config configs/qt_config.json
```
The configuration file (qt_config.json) should specify the necessary parameters for training.

#### **Evaluating a QT Model**

To evaluate a trained QT model, use the evaluate_qt.py script:

```bash
python scripts/evaluate_qt.py --model_path models/qt_model.pth --config configs/qt_config.json
```

## Example Notebooks

The /notebooks directory contains Jupyter notebooks for exploration and experimentation. These notebooks provide examples of how to use the models and layers in the project.

- `exploration.ipynb`: Initial data exploration and preprocessing
- `experimentation.ipynb`: Experimentation with different model architectures and training settings

## Documentation

Detailed documentation for each component is available in the `/docs` directory. Key files include:

- `qnn_overview.md`: Overview of Quantum Neural Networks
- `qt_overview.md`: Overview of Quantum Transformers
- `api_reference.md`: API reference for the project's modules and functions

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file in the `/docs` directory for guidelines on how to contribute to the project.

## Support

If you encounter any issues or have questions, please open an issue on the project's GitHub repository.