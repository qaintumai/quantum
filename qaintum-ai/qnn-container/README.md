# QNN: Quantum Neural Networks Framework
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-%3E=3.8-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.29.1-green)
![Torch](https://img.shields.io/badge/PyTorch-2.2.2-red)

## **Overview**
This folder contains the code for Quantum Neural Networks (QNN) based on the Continuous Variable (CV) model of Quantum Computing implemented using Quantum Optics.

Quantum computing is a method of computation that utilizes physical mediums operating based on the principles of quantum mechanics. There are two types of quantum computing:
* Discrete Variable (DV) Quantum Computing: quantizing digital computing based on the binary logic. The basis states are |0> and |1>.
* Continuous Variable Quantum Computing: using the continuous properties of nature. The basis states are |0>, |1>, ..., |n>

The difference between DV quantum computing and CV quantum computing is stated in this paper: [Quantum computing overview](https://arxiv.org/pdf/2206.07246).

The actual implementation of CV quantum computing was realized by Xanadu in 2020 using quantum optics. The architecture of the Photonic CV chip implemented by Xanadu can be found in this [paper](https://arxiv.org/abs/2103.02109).

CV Quantum Neural Networks (QNNs) is a framework for quantum-enhanced machine learning, designed to integrate quantum computing with classical deep learning models. Photonic QNN faithfully implements classical neural networks due to optical gates capable of representing bias addition and nonlinear activation function. This framework provides modular components for building, training, and deploying hybrid quantum-classical neural networks.

## **Key Features**
- 🧠 **Quantum Model**: Leverages quantum circuits for feature encoding and QNN layer processing.
- 🔥 **PyTorch & PennyLane Integration**: Seamlessly works with **PyTorch** for training and **PennyLane** for quantum circuit simulations.
- ⚡ **Custom Quantum Layers**: Provides pre-built layers like **Quantum Variational Circuits** and **Quantum Convolutional Networks**.
- 📡 **Supports Multiple Quantum Devices**: Compatible with simulators and real quantum hardware (Xanadu's X8) via Pennylane.

## **Installation**
### **1. Install via `pip` (Coming Soon)**
Once published on PyPI, you can install QNN using:
```bash
pip install qnn
