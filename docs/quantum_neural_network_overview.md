# Analog Quantum Neural Network

This implementation of quantum neural networks (QNN) utilizes the Continuous-Variable (CV) model of Quantum Computing. 
One key advantage of quantum computing is its inherent parallel processing capability, enabled by the superposition 
property of computational basis states. This advantage is further enhanced in Analog (CV) Quantum Computing (AQC), 
which offers a higher number of basis states compared to Digital (qubit-based or discrete) Quantum Computing (DQC).

The Quantum Neural Network (QNN) architecture used in this repository is based on 
[Continuous variable quantum neural networks](https://arxiv.org/pdf/1806.06871v1.pdf). The key features in this 
architecture implemented in Quantum Optics is the presense of the displacement property for implementing bias addition 
and the nonlinear property for implementing nonlinear activation functions, allowing for the true definition of 
"deep learning" in quantum.

The dataflow of this architecture is composed of:
- data encoding: converting classical data into quantum states by using input entries as parameters of quantum gates
- quantum circuit: composed of parametric (variational) quantum gates whose optimal parameters are learned during training
- optimization: parameter updates calculated on classical processing units.
  
It is heuristically observed that QNNs have the following advantages:
- substantial reduction of parameters to train
- faster convergence.

Continuous Variable Quantum Neural Networks (CV-QNNs) leverage the principles of continuous variable quantum computation, which is a distinct paradigm from the more commonly discussed qubit-based (discrete variable) quantum computing. Continuous variable quantum computation uses quantum states that are described by continuous variables, such as the position and momentum of quantum harmonic oscillators.

Key Concepts of Continuous Variable Quantum Neural Networks
Quantum Harmonic Oscillators:

CV quantum systems typically use modes of the electromagnetic field, which can be described by quantum harmonic oscillators. The states of these oscillators are characterized by continuous variables.
Quantum States and Operations:

Quantum States: In CV quantum systems, the quantum states are represented in an infinite-dimensional Hilbert space. Common states include Gaussian states such as coherent states, squeezed states, and thermal states.
Quantum Operations: Operations on CV quantum states include Gaussian operations (such as beam splitters and squeezers) and non-Gaussian operations (such as cubic phase gates).
Quantum Neural Network Architecture:

Quantum Layers: A CV-QNN consists of layers of quantum operations. Each layer can be composed of a sequence of Gaussian and non-Gaussian operations.
Parameterization: Similar to classical neural networks, CV-QNNs have parameters that can be optimized during training. These parameters can be the displacement, squeezing parameters, rotation angles, and interaction strengths in the quantum gates.
Measurement: The output of a CV-QNN is obtained by measuring the quantum state at the end of the network. Homodyne or heterodyne detection methods are commonly used to measure continuous variables.
Training and Optimization:

Cost Function: The training of a CV-QNN involves defining a cost function that measures the difference between the desired output and the actual output of the network.
Optimization Algorithms: Gradient-based optimization methods are often used to update the parameters of the quantum gates. This process can involve techniques like backpropagation adapted for quantum circuits.


### Words of Caution
Due to the decorator in Pennylane for quantum circuits, the number of wires need to be defined for data_encoding and qnn_layer, but cannot be passed as arguments. So we are going through an annoying process of manually defining them inside those functions in src/models/qnn_layer. Also inside src/models/qnn_model.py, the device need to be defined according to the number of wires and the number of bases which are manually defined for now. Future topic of your contribution.
  
