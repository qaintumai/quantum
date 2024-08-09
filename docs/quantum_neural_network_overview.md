# Analog Quantum Neural Network

This implementation of quantum neural networks (QNN) utilizes the Continuous-Variable (CV) model of Quantum Computing (In this repo, the term Analog Quantum Computing will be used to mean CV Model of Quantum Compuing).
One key advantage of quantum computing is its inherent parallel processing capability, enabled by the superposition
property of computational basis states. This advantage is further enhanced in Analog (CV) Quantum Computing (AQC),
which offers a higher number of basis states compared to Digital (qubit-based or discrete) Quantum Computing (DQC).

The Quantum Neural Network (QNN) architecture used in this repository is based on
[Continuous variable quantum neural networks](https://arxiv.org/pdf/1806.06871v1.pdf) developed by Killoran et al. The key features in this
architecture implemented in Quantum Optics is the presense of the displacement property for implementing bias addition
and the nonlinear property for implementing nonlinear activation functions, allowing for the true definition of
"deep learning" in quantum.

The dataflow of this architecture is composed of:
- quantum data encoding: converting classical data into quantum states by using input entries as parameters of quantum gates
- quantum circuit: composed of parametric (variational) quantum gates whose optimal parameters are learned during training
- optimization: parameter updates calculated on classical processing units.

It is heuristically observed that QNNs have the following advantages:
- substantial reduction of parameters to train
- faster convergence.

Analog Quantum Neural Networks (A-QNNs) leverage the principles of CV quantum computation, which is a distinct paradigm from the more commonly discussed qubit-based (discrete variable AKA digital using the binary system) quantum computing. AQC uses quantum states that are described by continuous or analog variables, such as the position and momentum of quantum harmonic oscillators.

Key Concepts of A-QNNs
Quantum Harmonic Oscillators:

Analog quantum systems typically use modes of the electromagnetic field, which can be described by quantum harmonic oscillators. The states of these oscillators are characterized by continuous variables (analog).
Quantum States and Operations:

Quantum States: In analog quantum systems, the quantum states are represented in an infinite-dimensional Hilbert space. The states used for our algorithm are squeezed Gaussian states while other states include coherent states, squeezed states, and thermal states.
Quantum Operations: Operations on analog quantum states include Gaussian operations (such as beam splitters and squeezers) and non-Gaussian operations (such as cubic phase gates).
Quantum Neural Network Architecture:

Quantum Layers: A A-QNN consists of layers of quantum operations. Each layer can be composed of a sequence of Gaussian and non-Gaussian operations.
Parameterization: Similar to classical neural networks, A-QNNs have parameters that can be optimized during training. These parameters can be the displacement, squeezing parameters, rotation angles, and interaction strengths in the quantum gates.
Measurement: The output of a A-QNN is obtained by measuring the quantum state at the end of the network. In our algorithm using Xanadu's X8, measurements are performed by counting the number of photons per quantum mode (waveguides of quantum light).
Training and Optimization:
The parameters controlling the quantum circuit are updated during the training process using the classical optimization techniques on classical computers.
Cost Function: The training of a A-QNN involves defining a cost function that measures the difference between the desired output and the actual output of the network.
Optimization Algorithms: Gradient-based optimization methods are often used to update the parameters of the quantum gates. This process can involve techniques like backpropagation adapted for quantum circuits.


### Words of Caution
Due to the decorator in Pennylane for quantum circuits, the number of wires need to be defined for quantum_data_encoder and qnn_layer, but cannot be passed as arguments. So we are going through an annoying process of manually defining them inside those functions in src/models/qnn_layer. Also inside src/models/qnn_model.py, the device need to be defined according to the number of wires and the number of bases which are manually defined for now. Future topic of your contribution.

