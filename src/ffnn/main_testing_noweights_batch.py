from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

X = np.array([[0.05, 0.10]])
target = np.array([[0.01, 0.99]])

model = FFNN.FFNN([2, 2, 2], X, target, 0.5)
model.setActivationUniform(Activation.tanh)
model.initializeWeightRandomUniform(-1, 1)

model.train(batch_size=32, learning_rate=model.learning_rate, epochs=10, printResults=True)