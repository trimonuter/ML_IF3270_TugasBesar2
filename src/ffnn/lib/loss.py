import numpy as np
from lib import activation as Activation

# MSE
def mse(target, prediction):
    rows = target.shape[0]
    return np.sum((target - prediction) ** 2) / rows

# Binary Cross Entropy
def bce(target, prediction):
    epsilon = 1e-12 # avoiding log(0) errors
    prediction = np.clip(prediction, epsilon, 1 - epsilon)

    rows = target.shape[0]
    return -(np.sum(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)) / rows)

# Categorical Cross Entropy
def cce(target, prediction):
    epsilon = 1e-12 # avoiding log(0) errors    
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    
    rows = target.shape[0]
    return -(np.sum(target * np.log(prediction)) / rows)

def categorical_cross_entropy_gradient(X, y_true, W):
    input = np.dot(X, W)  # Output sebelum softmax
    y_pred = Activation.softmax(input)  # Softmax probabilities
    dL_dy = (y_pred - y_true) / X.shape[0]  
    dW = np.dot(X.T, dL_dy)
    return dW

def getErrorDerivativeMatrix(loss, target, prediction):
    if loss == mse:
        return 2 * (target - prediction)
    elif loss == bce:
        return (prediction - target) / (prediction * (1 - prediction))
    elif loss == cce:
        epsilon = 1e-12
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        return target / prediction
    else:
        raise ValueError("Loss function not recognized")