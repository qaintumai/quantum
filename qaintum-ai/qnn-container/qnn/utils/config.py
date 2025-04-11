# Copyright 2025 The qAIntum.ai Authors. All Rights Reserved.
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

def get_qnn_config(
    num_wires=4,  # Default: 4 wires
    num_basis=5,  # Default: cutoff dimension of 5
    output_size="single",  # Default: single
):
    """
    Returns a dictionary containing the configuration for the Quantum Neural Network (QNN).

    Parameters:
    - num_wires (int): Number of quantum wires.
    - num_basis (int): Cutoff dimension for the quantum device.
    - output_size (string): Whether to return a single output, multiple outputs, or probabilities.

    Returns:
    - dict: A dictionary containing the QNN configuration.
    """
    if num_wires <= 0:
        raise ValueError("num_wires must be greater than 0.")
    if num_basis <= 0:
        raise ValueError("num_basis must be greater than 0.")

    return {
        "num_wires": num_wires,
        "num_basis": num_basis,
        "output_size": output_size,
    }