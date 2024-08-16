# Copyright 2024 The qAIntum.ai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Quantum Transformer Example

This script demonstrates the training and evaluation of a Quantum Transformer for sentiment analysis on IMDB reviews dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
import pennylane as qml
import sys
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Add the src directory to the Python path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from layers.input_embedding import InputEmbedding
from layers.scaled_dot_product import ScaledDotProduct
from layers.multi_headed_attention import MultiHeadedAttention
from layers.quantum_data_encoder import QuantumDataEncoder
from layers.quantum_layer import QuantumNeuralNetworkLayer
from layers.weight_initializer import WeightInitializer
from layers.qnn_circuit import qnn_circuit
from models.quantum_neural_network import QuantumNeuralNetworkModel
#from layers.quantum_feed_forward import QuantumFeedForward
from models.quantum_encoder import QuantumEncoder
from models.quantum_decoder import QuantumDecoder
from models.quantum_transformer import Transformer

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the IMDb dataset
train_iter, test_iter = IMDB()

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Function to process raw text into tensor
def text_pipeline(text):
    return vocab(tokenizer(text))

# Function to process labels into tensor
def label_pipeline(label):
    return 1 if label == 'pos' else 0

# Pad sequences to a fixed length
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    # Pad sequences to the same length
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab['<pad>'])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list.to(device), label_list.to(device)

# Create DataLoaders
train_iter, test_iter = IMDB()
train_dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=8, shuffle=False, collate_fn=collate_batch)

# Define parameters
num_encoder_layers = 6
num_decoder_layers = 6
embed_len = 64
num_heads = 8
seq_len = 20
batch_size = 32
vocab_size = 100
dropout = 0.1
device = 'cpu'

model=Transformer(num_encoder_layers, num_decoder_layers, embed_len, num_heads, batch_size, vocab_size, dropout, device)  # Have to mention parameters

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress updates
def train_model(model, train_dataloader, criterion, optimizer, num_epochs=1):
    model.train()
    total_steps = len(train_dataloader) * num_epochs
    step = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        
        print(f"Epoch {epoch + 1}/{num_epochs} started")
        
        for batch_idx, (text, labels) in enumerate(train_dataloader):
            batch_start_time = time.time()
            
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            # Print progress within the epoch
            if batch_idx % 10 == 0 or batch_idx == len(train_dataloader) - 1:
                elapsed = time.time() - batch_start_time
                print(f"  Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f} - Elapsed: {elapsed:.2f}s")
            
            # Print overall progress
            overall_progress = (step / total_steps) * 100
            print(f"  Overall Progress: {overall_progress:.2f}% completed")
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s with average loss {total_loss/len(train_dataloader):.4f}")
        print("------------------------------------------------------")
        
    print("Training completed.")

# Train the model with the updated interactive logging
train_model(model, train_dataloader, criterion, optimizer)

# Evaluation function
def evaluate_model(model, test_dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for text, labels in test_dataloader:
            outputs = model(text)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

evaluate_model(model, test_dataloader)
