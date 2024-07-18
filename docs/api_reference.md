API Reference
Table of Contents

Introduction
Modules
Quantum Neural Networks
Quantum Transformers
Common Utilities
Quantum Neural Networks API
qnn_model.py
qnn_utils.py
Quantum Transformers API
transformer.py
data_encoding.py
decoder_block.py
encoder_block.py
feed_forward_block.py
input_embedding.py
multi_headed_attention.py
quantum_layer.py
scaled_dot_product.py
weight_initializer.py
Common Utilities API
data_loader.py
config.py
Introduction
This document provides a detailed reference for the APIs available in the Quantum Project. Each module and its respective components are described in detail, including the available functions, their parameters, return types, and usage examples.

Modules
Quantum Neural Networks

The Quantum Neural Networks module contains the core components for building and training quantum neural network models.

Quantum Transformers

The Quantum Transformers module provides the implementation of transformer models adapted for quantum computations.

Common Utilities

The Common Utilities module contains shared utilities used across the project, including data loading and configuration management.

Quantum Neural Networks API
qnn_model.py

Class: QuantumNeuralNetwork

Description: A class representing a quantum neural network model.
Methods:
__init__(self, config): Initializes the QNN with the given configuration.
Parameters:
config (dict): Configuration dictionary for the QNN.
train(self, data, labels): Trains the QNN on the provided data and labels.
Parameters:
data (array-like): Training data.
labels (array-like): Training labels.
predict(self, data): Predicts labels for the provided data.
Parameters:
data (array-like): Data to predict labels for.
Returns:
predictions (array-like): Predicted labels.
qnn_utils.py

Function: preprocess_data

Description: Preprocesses the data for use in the QNN.
Parameters:
data (array-like): Data to be preprocessed.
Returns:
processed_data (array-like): Preprocessed data.
Quantum Transformers API
transformer.py

Class: QuantumTransformer

Description: A class representing a quantum transformer model.
Methods:
__init__(self, config): Initializes the transformer with the given configuration.
Parameters:
config (dict): Configuration dictionary for the transformer.
encode(self, data): Encodes the input data.
Parameters:
data (array-like): Data to be encoded.
Returns:
encoded_data (array-like): Encoded data.
data_encoding.py

Function: encode_data

Description: Encodes data for use in quantum transformers.
Parameters:
data (array-like): Data to be encoded.
Returns:
encoded_data (array-like): Encoded data.
decoder_block.py

Class: DecoderBlock

Description: A class representing a decoder block in the transformer.
Methods:
__init__(self, config): Initializes the decoder block with the given configuration.
Parameters:
config (dict): Configuration dictionary for the decoder block.
encoder_block.py

Class: EncoderBlock

Description: A class representing an encoder block in the transformer.
Methods:
__init__(self, config): Initializes the encoder block with the given configuration.
Parameters:
config (dict): Configuration dictionary for the encoder block.
feed_forward_block.py

Class: FeedForwardBlock

Description: A class representing a feed-forward block in the transformer.
Methods:
__init__(self, config): Initializes the feed-forward block with the given configuration.
Parameters:
config (dict): Configuration dictionary for the feed-forward block.
input_embedding.py

Class: InputEmbedding

Description: A class representing the input embedding layer in the transformer.
Methods:
__init__(self, config): Initializes the input embedding with the given configuration.
Parameters:
config (dict): Configuration dictionary for the input embedding.
multi_headed_attention.py

Class: MultiHeadedAttention

Description: A class representing the multi-headed attention mechanism in the transformer.
Methods:
__init__(self, config): Initializes the multi-headed attention with the given configuration.
Parameters:
config (dict): Configuration dictionary for the multi-headed attention.
quantum_layer.py

Class: QuantumLayer

Description: A class representing a quantum layer in the transformer.
Methods:
__init__(self, config): Initializes the quantum layer with the given configuration.
Parameters:
config (dict): Configuration dictionary for the quantum layer.
scaled_dot_product.py

Function: scaled_dot_product_attention

Description: Computes scaled dot-product attention.
Parameters:
queries (array-like): Query vectors.
keys (array-like): Key vectors.
values (array-like): Value vectors.
Returns:
attention_output (array-like): Output of the attention mechanism.
weight_initializer.py

Function: initialize_weights

Description: Initializes weights for the transformer.
Parameters:
shape (tuple): Shape of the weights.
Returns:
weights (array-like): Initialized weights.
Common Utilities API
data_loader.py

Function: load_data

Description: Loads data from the specified source.
Parameters:
source (str): Path or URL to the data source.
Returns:
data (array-like): Loaded data.
config.py

Function: get_config

Description: Retrieves the configuration settings.
Parameters:
config_file (str): Path to the configuration file.
Returns:
config (dict): Configuration dictionary.
