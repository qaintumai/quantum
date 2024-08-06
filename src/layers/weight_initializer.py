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

import numpy as np
#TODO: Define the WeightInitializer class


class WeightInitializer:
    @staticmethod
    def init_weights(layers, num_wires, active_sd=0.0001, passive_sd=0.1):
        M = (num_wires - 1) * 2 + num_wires  # Number of interferometer parameters

        int1_weights = np.random.normal(size=[layers, M], scale=passive_sd)
        s_weights = np.random.normal(size=[layers, num_wires], scale=active_sd)
        int2_weights = np.random.normal(size=[layers, M], scale=passive_sd)
        dr_weights = np.random.normal(size=[layers, num_wires], scale=active_sd)
        k_weights = np.random.normal(size=[layers, num_wires], scale=active_sd)

        weights = np.concatenate(
            [int1_weights, s_weights, int2_weights, dr_weights, k_weights], axis=1)

        return weights
