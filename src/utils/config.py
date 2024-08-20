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

#Variables to access across project . . .
import torch

num_wires = 8
num_layers = 2
num_basis = 2
single_output = True
multi_output = False
probabilities = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_device():
    return device

def set_config_variable(name, value):
    """
    Set the value of a configuration variable by name.

    Args:
        name (str): The name of the configuration variable.
        value (any): The value to set the configuration variable to.
    """
    global num_wires, num_layers, num_basis, single_output, multi_output, probabilities, device
    if name in globals():
        globals()[name] = value
    else:
        raise ValueError(f"Config variable '{name}' does not exist.")

# Example usage:
# set_config_variable('single_output', True)
# set_config_variable('num_layers', 6)
