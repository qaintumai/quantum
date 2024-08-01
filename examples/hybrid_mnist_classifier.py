import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# import tensorflow as tf
# import tensorflow_datasets as tfds
# from tensorflow import keras
# from tensorflow.keras import layers
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)


from models.quantum_neural_network import QuantumNeuralNetworkModel
from layers.quantum_data_encoder import QuantumDataEncoder
from layers.weight_initializer import WeightInitializer
from layers.qnn_circuit import qnn_circuit
from layers.quantum_layer import QuantumNeuralNetworkLayer

### PREPROCESSING ###

#Load MNIST Data Set (replace with data loading)
mnist = keras.datasets.mnist

# datasets are numpy.ndarrays
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()     

# normalize the image data to pixel values from 0 ~ 255 to 0 ~ 1
X_train, X_test = X_train / 255.0, X_test / 255.0


#One hot encoding, necessary for file
def one_hot(labels):  
       
    depth =  2**4                       # 10 classes + 6 zeros for padding
    indices = labels.astype(np.int32)    
    one_hot_labels = np.eye(depth)[indices].astype(np.float32) 
    
    return one_hot_labels

# one-hot encoded labels, each label of length cutoff dimension**2
y_train, y_test = one_hot(Y_train), one_hot(Y_test)

# using only 600 samples for training in this experiment
n_samples = 600
test_samples = 100
X_train, X_test, y_train, y_test = X_train[:n_samples], X_test[:test_samples], y_train[:n_samples], y_test[:test_samples]

keras.backend.set_floatx('float32')

## CREATE CLASSICAL MODEL ##
## update with torch instead? ##
# model = keras.models.Sequential([
#                                  layers.Flatten(input_shape = (28,28)),
#                                  layers.Dense(392, activation ="elu"),
#                                  layers.Dense(196, activation ="elu"),
#                                  layers.Dense(98),
#                                  layers.Dense(49, activation ="elu"),
#                                  layers.Dense(30),
#                                 ])

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Flatten(),  # Flatten the input
    nn.Linear(28 * 28, 392),  # Dense layer with 392 units
    nn.ELU(),  # ELU activation function
    nn.Linear(392, 196),  # Dense layer with 196 units
    nn.ELU(),  # ELU activation function
    nn.Linear(196, 98),  # Dense layer with 98 units
    nn.Linear(98, 49),  # Dense layer with 49 units
    nn.ELU(),  # ELU activation function
    nn.Linear(49, 30)  # Dense layer with 30 units
)

# More than a million parameters for the classical circuit
model.summary()
num_wires = 4
encoder = QuantumDataEncoder(num_wires) # 4 qumodes
encoder.encode(X_train)
num_modes = 4
cutoff_dim = 2

# select a devide 
dev = qml.device("strawberryfields.fock", wires=num_modes, cutoff_dim=cutoff_dim) 

@qml.qnode(dev, interface="tf")
def quantum_nn(inputs, var):
    # Encode input x into quantum state
    encode_data(inputs)

    # iterative quantum layers
    for v in var:
        layer(v)

    return qml.probs(wires=[0, 1, 2, 3])  # Measurement

weight_shape = {'var': (4,32)}          # 4 layers and 32 parameters per layer, Keras layer will initialize.

num_layers = 4
qlayer = qml.qnn.KerasLayer(quantum_nn, weight_shape, output_dim = 4)

# add to the classical sequential model
model.add(qlayer)

lr = 0.02
opt = keras.optimizers.SGD(lr)
model.compile(opt, loss = 'categorical_crossentropy', metrics =['accuracy'])

hybrid = model.fit(X_train, 
                   y_train,
                   epochs = 70,
                   batch_size = 64,
                   shuffle = True, 
                   validation_data = (X_test, y_test))

model.summary()


plt.title('model loss')
plt.plot(hybrid.history['loss'], '-g')
plt.ylabel('loss')
plt.show()

plt.title('model accuracy')
plt.plot(hybrid.history['accuracy'], '-g')
plt.ylabel('accuracy')
plt.show()