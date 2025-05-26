from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

network = [
    np.array(
        [[0.35, 0.35],
        [0.15, 0.25],
        [0.20, 0.30]]),
    np.array(
        [[0.60, 0.60],
        [0.40, 0.50],
        [0.45, 0.55]]),
]
X = np.array([[0.05, 0.10]])
target = np.array([[0.01, 0.99]])

model = FFNN.FFNN([2, 2, 2], X, target, X, target, 0.5)
model.setActivationUniform(Activation.linear)
model.setWeights(network)

model.train(batch_size=32, learning_rate=model.learning_rate, epochs=10, printResults=True)