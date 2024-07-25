## **API Reference**

### **Table of Contents**

1. Introduction
2. Modules
   * Layers
   * Models
   * Utilities
3. Layers API
   * input_embedding.py
   * multi_headed_attention.py
   * qnn_multi_output.py
   * qnn_probabilities.py
   * qnn_single_output.py
   * quantum_data_encoding.py
   * quantum_feed_forward.py
   * quantum_layer.py
   * scaled_dot_product.py
   * weight_initializer.py
4. Models API
   * quantum_decoder.py
   * quantum_encoder.py
   * quantum_neural_network.py
   * quantum_transformer.py
5. Utilities API


### Introduction
This document provides a detailed reference for the APIs available in the Quantum Project. Each module and its respective components are described in detail, including the available functions, their parameters, return types, and usage examples.

### Modules

#### Layers

The Layers module contains the core components for building various layers in quantum neural networks and transformers.

For qauntum neural networks (qnn), the following components are needed:
* quantum data encoding
* qnn layer: this is a quantum version of a classical layer composed of weight matrix, bias addition, and nonlinear activation function.
* weight initializer: this creates a randomly initialized vector to be used as parameters of the quantum gates of the quantum layers.
* qnn circuit: this process builds a circuit composed of quantum data encoding and a quantum neural network. Depending of the desired output, there are three methods we can use:
  * single_output: returns a single value as a result of the quantum computation using the expected value measurement method applied to the first wire.
  * multi_output: returns a vector of multiple values equal to the number of wires used. 
  * probabilities: the size of the output is equal to the number of basis raised to the power of the number of wires.
 
For quantum transformers, most of the components are the same as classical transformers: input embedding, scaled dot product, and multi-headed attention. The only component that is different is the feed forward block, which is replaced with a quantum neural network.


#### Models

The Models module provides the implementation of quantum neural networks and quantum transformers. Quantum encoders and quantum decoders are used to build quantum transformers.

#### Utilities

The Utilities module contains shared utilities used across the project.

### Layers API
#### input_embedding.py

##### Class: InputEmbedding

* Description: A class representing the input embedding layer.
* Methods:
    * __init__(self, config): Initializes the input embedding with the given configuration.
    * Parameters: input_vocab_size, embed_len, dropout=0.1, device='cpu'.
#### multi_headed_attention.py

##### Class: MultiHeadedAttention

* Description: A class representing the multi-headed attention mechanism.
* Methods:
  * __init__(self, config): Initializes the multi-headed attention with the given configuration.
  * Parameters: num_heads, embed_len, batch_size, mask=None.

#### qnn_circuit.py

##### Function: qnn_circuit

* Description: A function building a quantum neural network circuit. The measuremet outputs are of the shape:
              * single output: expectation value of the first wire.
              * multi output: expectation values of all the wires.
              * probabilities: output of size the number of basis states ^ the number of wires.
* Parameters: input data, initialized weights.

  
#### quantum_data_encoding.py

##### Class: QuantumDataEncoder

* Description: A class for encoding data for quantum computations.
* Methods:
  * __init__(self, config): Initializes the data encoding with the given configuration.
  * Parameters:
config (dict): Configuration dictionary for data encoding.

#### quantum_feed_forward.py

##### Class: QuantumFeedForward

* Description: A class representing a feed-forward layer in quantum neural networks.
* Methods:
__init__(self, config): Initializes the feed-forward layer with the given configuration.
Parameters:
config (dict): Configuration dictionary for the feed-forward layer.

#### qnn_layer.py

##### Class: QNNLayer

* Description: A class representing a generic quantum neural network layer.
* Methods:
__init__(self, config): Initializes the quantum layer with the given configuration.
Parameters:
config (dict): Configuration dictionary for the quantum layer.

#### scaled_dot_product.py

##### Class: scaled_dot_product_attention

* Description: Computes scaled dot-product attention.
* Parameters:
            * queries (array-like): Query vectors.
            * keys (array-like): Key vectors.
            * values (array-like): Value vectors.
* Returns: Output (array-like) of the attention mechanism.

#### weight_initializer.py

##### Class: initialize_weights

* Description: Initializes weights for layers.
* Parameters:
shape (tuple): Shape of the weights.
* Returns:
weights (array-like): Initialized weights.

### Models API
#### quantum_decoder.py

##### Class: QuantumDecoder

* Description: A class representing a quantum decoder model.
* Methods:
  * __init__(self, config): Initializes the decoder with the given configuration.
  * Parameters:
config (dict): Configuration dictionary for the decoder.

#### quantum_encoder.py

##### Class: QuantumEncoder

* Description: A class representing a quantum encoder model.
* Methods:
  * __init__(self, config): Initializes the encoder with the given configuration.
  * Parameters:
config (dict): Configuration dictionary for the encoder.

#### quantum_neural_network.py

##### Class: QuantumNeuralNetwork

##### Description: A class representing a quantum neural network model.
* Methods:
  * __init__(self, config): Initializes the QNN with the given configuration.
    * Parameters:
config (dict): Configuration dictionary for the QNN.
  * train(self, data, labels): Trains the QNN on the provided data and labels.
    * Parameters:
      * data (array-like): Training data.
      * labels (array-like): Training labels.
  * predict(self, data): Predicts labels for the provided data.
    * Parameters:
    * data (array-like): Data to predict labels for.
    * Returns:
predictions (array-like): Predicted labels.

#### quantum_transformer.py

##### Class: QuantumTransformer

* Description: A class representing a quantum transformer model.
* Methods:
  * __init__(self, config): Initializes the transformer with the given configuration.
    * Parameters: config (dict): Configuration dictionary for the transformer.
  * encode(self, data): Encodes the input data.
    * Parameters:
data (array-like): Data to be encoded.
    * Returns:
encoded_data (array-like): Encoded data.

### Utilities API
#### data_loader.py

##### Function: load_data

* Description: Loads data from the specified source.
* Parameters:
source (str): Path or URL to the data source.
* Returns:
data (array-like): Loaded data.

#### config.py

##### Function: get_config

* Description: Retrieves the configuration settings.
* Parameters:
config_file (str): Path to the configuration file.
* Returns:
config (dict): Configuration dictionary.
