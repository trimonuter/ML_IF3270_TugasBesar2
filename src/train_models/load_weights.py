import numpy as np

weights = np.load('nusax_rnn_weights.npz')

with open('nusax_rnn_weights.txt', 'w') as f:
    for i, weight in enumerate(weights.files):
        f.write(f"Weight {i}:\n")
        f.write(str(weights[weight]) + "\n\n")