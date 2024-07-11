def qnn_layer(v):
    """

    """
    num_params = len(v)
    num_wires = 8

    # Interferometer 1
    for i in range(num_wires - 1):
        idx = i * 2
        if idx + 1 < num_params:
            theta = v[idx]
            phi = v[idx + 1]
            qml.Beamsplitter(theta, phi, wires=[i % num_wires, (i + 1) % num_wires])

    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + i
        if idx < num_params:
            qml.Rotation(v[idx], wires=i)

    # Squeezers
    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + i
        if idx < num_params:
            qml.Squeezing(v[idx], 0.0, wires=i)

    # Interferometer 2
    for i in range(num_wires - 1):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + i * 2
        if idx + 1 < num_params:
            theta = v[idx]
            phi = v[idx + 1]
            qml.Beamsplitter(theta, phi, wires=[i % num_wires, (i + 1) % num_wires])

    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + (num_wires - 1) * 2 + i
        if idx < num_params:
            qml.Rotation(v[idx], wires=i)

    # Bias addition
    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + (num_wires - 1) * 2 + num_wires + i
        if idx < num_params:
            qml.Displacement(v[idx], 0.0, wires=i)

    # Non-linear activation function
    for i in range(num_wires):
        idx = (num_wires - 1) * 2 + num_wires + num_wires + (num_wires - 1) * 2 + num_wires + num_wires + i
        if idx < num_params:
            qml.Kerr(v[idx], wires=i)
