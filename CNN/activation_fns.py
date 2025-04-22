import numpy as np
from network_code.activation import Activation

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def softmax(x):
    x = np.clip(x, -1e3, 1e3)  # Prevent large exponents
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)

    # Avoid divide-by-zero
    sum_e_x = np.where(sum_e_x == 0, 1e-8, sum_e_x)

    return e_x / sum_e_x

def softmax_prime(x):
    # Usually handled in loss, but can use for completeness
    s = softmax(x)
    return s * (1 - s)

def softmax_derivative(x):
    return x  # Identity function since derivative is handled in loss