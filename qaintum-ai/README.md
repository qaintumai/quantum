# QAI: Quantum AI Framework
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-%3E=3.8-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.29.1-green)
![Torch](https://img.shields.io/badge/PyTorch-2.2.2-red)

## **Overview**
This folder contains the code for Quantum AI Quantum Neural Networks (QNN) based on the Continuous Variable (CV) model of Quantum Computing implemented using Quantum Optics. This framework provides modular components for building, training, and deploying Quantum Neural Networks (QNNs) for classification and regression tasks and Quantum Transformers (QTs) for sequential data generation tasks for Generative AI.

Quantum computing is a method of computation that utilizes physical mediums operating based on the principles of quantum mechanics. There are two types of quantum computing:
* Discrete Variable (DV) Quantum Computing: quantizing digital computing based on the binary logic. The basis states are |0> and |1>.
* Continuous Variable Quantum Computing: using the continuous properties of nature. The basis states are |0>, |1>, ..., |n>

The difference between DV quantum computing and CV quantum computing is stated in this paper: [Quantum computing overview](https://arxiv.org/pdf/2206.07246).

The implementation of CV quantum computing was pioneered by Xanadu in 2020 using quantum optics. The architecture of the Photonic CV chip is detailed in [paper](https://arxiv.org/abs/2103.02109).

## **Components**
- **Quantum Neural Network (QNN)**:
CV Quantum Neural Networks (QNNs) is a framework for quantum-enhanced machine learning, designed to integrate quantum computing with classical deep learning models. Photonic QNN faithfully implements classical neural networks due to optical gates capable of representing bias addition and nonlinear activation function. The optical quantum gates for implementing a neural network layer:
* weight matrix: interferometer + squeezing + interferometer
* bias addition: displacement
* nonlinear activation function: Kerr effect
is outline in [paper](https://arxiv.org/abs/1806.06871).
QNNs require a substantially reduced number of parameters to train due to its inherent parallelism from the superposition property of quantu mechanics.
- **Quantum Transformer (QT)**:
Quantum Transformers (QTs) extend transformer-based architectures by replacing the feedforward layers with QNNs. Key components of a transformer include:

**Input Encoding: Input embeddings with positional encoding for contextual understanding.

Multi-Headed Attention: Query, Key, and Value mechanisms.

*Feedforward Layer: Classical neural network, replaced by QNN for improved efficiency and parameter reduction.

- **Quantum Small Language Model (QSML) (Coming Soon)**:
QSML is designed as a compact, enterprise-grade language model leveraging quantum-enhanced transformers. Unlike traditional large language models, QSML utilizes optimized transformer blocks combined with quantum-enhanced learning techniques to provide efficient, domain-specific AI solutions.

## **Key Features**
- 🧠 **Quantum Models**: Leverages quantum circuits for feature encoding and processing.
- 🔥 **PyTorch & PennyLane Integration**: Seamlessly works with **PyTorch** for training and **PennyLane** for quantum circuit simulations.
- ⚡ **Custom Quantum Layers**: Provides pre-built layers like **Quantum Parametrized Circuits** with parameterized quantum gates whose parameters are learned through training.
- 📡 **Supports Multiple Quantum Devices**: Compatible with simulators and real quantum hardware (Xanadu's X8) via Pennylane.
