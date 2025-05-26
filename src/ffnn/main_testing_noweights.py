from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
from lib import loss as Loss
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# Load data from OpenML
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Split into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-Hot Encoding: Convert labels to one-hot vectors
from sklearn.preprocessing import OneHotEncoder

y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))
y_val = OneHotEncoder(sparse_output=False).fit_transform(y_val.reshape(-1, 1))

base_hidden_layer = [784, 128, 64, 10]
base_activation = Activation.tanh
base_learning_rate = 0.01
base_epochs = 20
# base_weight_init = .initializeWeightRandomUniform(-1, 1)
base_batch_size = 32

linear_model = FFNN.FFNN(base_hidden_layer, X_train, y_train, X_val, y_val, base_learning_rate)
linear_model.setActivationUniform(Activation.linear)
linear_model.initializeWeightRandomUniform(-1, 1)

linear_model_train_loss, linear_model_val_loss = linear_model.train(batch_size=base_batch_size, learning_rate=base_learning_rate, epochs=base_epochs)