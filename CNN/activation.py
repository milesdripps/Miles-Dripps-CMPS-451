
import numpy as np
from network_code.layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, grad_output, learning_rate):
        grad_output = np.squeeze(grad_output, axis=-1) if grad_output.ndim == 3 else grad_output
        return grad_output * self.activation_prime(self.input)
