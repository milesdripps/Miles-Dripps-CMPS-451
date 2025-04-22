import numpy as np
from network_code.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.input = None

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        input_reshaped = input.reshape(batch_size, -1)
        return (np.dot(input_reshaped, self.weights.T) + self.biases.T)  # shape: (batch_size, output_size)

    def backward(self, grad_output, learning_rate):
        np.clip(grad_output, -1e5, 1e5, out=grad_output)
        batch_size = grad_output.shape[0]
        grad_output_reshaped = grad_output.reshape(batch_size, -1)
        input_reshaped = self.input.reshape(batch_size, -1)

        # Compute gradients
        grad_weights = np.dot(grad_output_reshaped.T, input_reshaped)
        grad_biases = np.sum(grad_output_reshaped, axis=0, keepdims=True).T
        grad_input = np.dot(grad_output_reshaped, self.weights)

        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input.reshape(self.input.shape)
