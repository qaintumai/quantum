def init_weights(layers, modes, active_sd=0.0001, passive_sd=0.1):
    M = (modes-1)*2 + modes  # Number of interferometer parameters

    int1_weights = np.random.normal(size=[layers, M], scale=passive_sd)
    s_weights = np.random.normal(size=[layers, modes], scale=active_sd)
    int2_weights = np.random.normal(size=[layers, M], scale=passive_sd)
    dr_weights = np.random.normal(size=[layers, modes], scale=active_sd)
    k_weights = np.random.normal(size=[layers, modes], scale=active_sd)

    weights = np.concatenate([int1_weights, s_weights, int2_weights, dr_weights, k_weights], axis=1)

    return weights
