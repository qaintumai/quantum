# Photonic Analog Quantum Computing
This folder contains the code for Quantum Neural Networks (QNN) and Quantum Transformers (QT) based on Photonic Analog Quantum Computing.

Quantum computing is a method of computation that utilizes physical mediums operating on the principles of quantum mechanics. There are two types of quantum computing:
* Digital Quantum Computing: quantizing digital computing based on the binary logic.
* Analog Quantum Computing: using the continuous (analog) properties of nature.

The difference between digital (discrete) quantum computing and analog (continuous) quantum computing is stated in this paper: [Quantum computing overview](https://arxiv.org/pdf/2206.07246).

The actual implementation of analog quantum computing (AQC) was realized by Xanadu in 2020 using quantum optics. The architecture of the Photonic AQC chip implemented by Xanadu can be found in this [paper](https://arxiv.org/abs/2103.02109).

Photonic Analog Quantum Computing has the following benefits:
* Compatible with the existing communications infrastructure.
* Operates at room-temperature.
* Higher dimensional computation space.
* Easy to network and multiplex
* Low cost of mass production
* Mountable on smartphones, laptops, and edge devices.

National Science Foundation (NSF) and Department of Energy (DOE) have been designing a blueprint for Quantum Internet on top of classical internet infrastructure as seen in these papers:
* [Development of Quantum InterConnects (QuICs) for Next-Generation Information Technologies, 2019](https://arxiv.org/pdf/1912.06642)
* [A Roadmap for Quantum Interconnects, 2022](https://publications.anl.gov/anlpubs/2022/12/179439.pdf)

Quantum InterConnects are composed of
* Quantum Communications
* Quantum Computing
* Quantum Memory
* Transducers
* Quantum Sensing.

Most of the research in these areas is based on quantum optics. Hence quantum computing using quantum optics will seemlessly integrate into the whole architecture of Quantum Internet.

## Getting Started

### Clone the Repository
```sh
git clone https://github.com/qaintumai/quantum.git
cd quantum
```

### Create a Branch
```sh
git checkout -b <new_branch_name>
```

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

```sh
pip install -r requirements.txt
```
* Caution: Sometimes, pip may default to a user installation outside the virtual environment instead of installing packages within the virtual environment's site-packages. To avoid this, you can run

```sh
/Users/<your_user_directory_name>/github/quantum/venv/bin/pip install -r requirements.txt
```
or 

```sh
/Users/<your_user_directory_name>/github/quantum/venv/bin/pip install --no-user -r requirements.txt
```

### Running

```sh
pip install -e .
./examples/qnn_binary_classifier.py
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

### Make Changes
Edit, add, or remove files as needed in your project. For example, you might edit a file called qnn_binary_classification.py.

### Stage the Changes
Add the files you changed to the staging area.
```sh
git add <directory_where_the_changed_file_is_located>/<file_name_with_changes>
```

If you made changes to multiple files, to stage all changes you can use:
```sh
git add .
```

### Commit the Changes
Commit the staged changes with a descriptive commit message.
```sh
git commit -m "Add changes to <file_name_with_changes>"
```

### Push the New Branch to 'quantum'
Push your new branch with the changes to the 'quantum' repository.
```sh
git push origin <new_branch_name>

```

### Contributing and Best Practices

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

#### Coming Soon

- PyPi
- Type checking with mypy
- Linting with flake8


