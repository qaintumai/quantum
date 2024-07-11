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
  
## Getting Started

### Virtual Environment
```shell
python3 -m venv venv
source venv/bin/activate
```

### Dependency Installation

#### Third-party Dependencies

- PyTorch
- Pennylane (**0.29.1**)
- Scikit Learn
- Pandas
- Numpy

```shell
pip install -r requirements.txt
```

### Running

```sh
pip install -e . 
./examples/torch_qnn.py
```

You should see the following output:

```sh
loss tensor(0.5000, grad_fn=<MseLossBackward0>)
Average loss over epoch 1: 0.0250
loss tensor(0.5781, grad_fn=<MseLossBackward0>)
Average loss over epoch 2: 0.0289
loss tensor(0.5312, grad_fn=<MseLossBackward0>)
Average loss over epoch 3: 0.0266
loss tensor(0.5312, grad_fn=<MseLossBackward0>)
Average loss over epoch 4: 0.0266
loss tensor(0.5000, grad_fn=<MseLossBackward0>)
Average loss over epoch 5: 0.0250
loss tensor(0.5625, grad_fn=<MseLossBackward0>)
Average loss over epoch 6: 0.0281
Accuracy: 85.0%
```

### Contributing and Best Practices

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

#### Coming Soon

- Hybrid Quantum/Classical Neural Network
- Type checking with mypy
- Linting with flake8


