import numpy as np
from network_code.layer import Layer


class BatchNormalization(Layer):
    def __init__(self, num_features):
        self.gamma = np.ones(num_features)  # Shape: (num_features,)
        self.beta = np.zeros(num_features)  # Shape: (num_features,)
        self.eps = 1e-5
        self.cache = None

    def forward(self, input):
        input = np.clip(input, -1e3, 1e3)

        # Input shape: (batch_size, num_features, height, width)
        batch_size, num_features, height, width = input.shape

        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)

        mean = np.mean(input, axis=(0, 2, 3), keepdims=True)
        var = np.var(input, axis=(0, 2, 3), keepdims=True)

        # Safe division to avoid zero division
        std_inv = 1.0 / np.sqrt(np.maximum(var, self.eps))

        self.normalized = (input - mean) * std_inv
        self.output = gamma * self.normalized + beta

        self.cache = (input, mean, var, std_inv)
        return self.output

    def backward(self, grad_output, learning_rate):
        np.clip(grad_output, -1e5, 1e5, out=grad_output)

        input, mean, var, std_inv = self.cache
        batch_size, num_features, height, width = input.shape

        gamma = self.gamma.reshape(1, -1, 1, 1)

        # Calculate gradients for gamma and beta
        d_gamma = np.sum(grad_output * self.normalized, axis=(0, 2, 3))
        d_beta = np.sum(grad_output, axis=(0, 2, 3))

        # Calculate gradients for input, variance, and mean
        dnormalized = grad_output * gamma

        # Gradient for variance (with safe division)
        dvar = np.sum(dnormalized * (input - mean) * -0.5 * std_inv ** 3, axis=(0, 2, 3), keepdims=True)

        # Gradient for mean
        dmean = np.sum(dnormalized * -std_inv, axis=(0, 2, 3), keepdims=True)

        # Clip gradients to avoid extreme values
        dvar = np.clip(dvar, -1e3, 1e3)
        dmean = np.clip(dmean, -1e3, 1e3)
        np.clip(d_gamma, -1e3, 1e3, out=d_gamma)
        np.clip(d_beta, -1e3, 1e3, out=d_beta)

        # Update gamma and beta
        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta

        # Calculate gradient for the input
        input_gradient = (
                dnormalized * std_inv
                + dvar * 2 * (input - mean) / (batch_size * height * width)
                + dmean / (batch_size * height * width)
        )

        return input_gradient
