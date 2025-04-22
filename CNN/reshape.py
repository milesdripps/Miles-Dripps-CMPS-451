import numpy as np
from network_code.layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape  # (channels, height, width)
        self.output_shape = output_shape  # (flat_size, 1)

    def forward(self, input):
        batch_size = input.shape[0]  # Preserve batch dimension
        return input.reshape(batch_size, *self.output_shape)  # (batch_size, flat_size, 1)

    def backward(self, grad_output, learning_rate):
        np.clip(grad_output, -1e5, 1e5, out=grad_output)
        return grad_output.reshape(grad_output.shape[0], *self.input_shape)  # (batch_size, channels, height, width)_gradient.reshape(self.input.shape)