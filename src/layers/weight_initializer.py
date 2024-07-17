import numpy as np
# Define the WeightInitializer class


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
