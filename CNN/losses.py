import numpy as np

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return (((1 - y_true) / (1 - y_pred)) - (y_true / y_pred)) / np.size(y_true)

def categorical_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze(axis=-1)
    return -np.sum(y_true * np.log(y_pred), axis=1)  # shape (batch_size,)

def categorical_crossentropy_gradient(y_true, y_pred):
    return y_pred - y_true  # already includes softmax derivative
