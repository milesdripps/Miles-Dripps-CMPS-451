import numpy as np
from scipy import signal
from network_code.layer import Layer


class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_shape = (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernel_shape = (depth, self.input_depth, kernel_size, kernel_size)
        scale = np.sqrt(2 / (self.input_depth * kernel_size * kernel_size))
        self.kernel_matrix = np.random.randn(*self.kernel_shape) * scale
        self.bias_matrix = np.zeros((depth, 1, 1))  # Single bias per filter
        self.input = None

    def forward(self, input):
        self.input = input

        # Handle both batch and single sample cases
        if input.ndim == 3:  # Single sample (C, H, W)
            input = input[np.newaxis, ...]  # Add batch dimension
            batch_size = 1
        else:  # Batch of samples (N, C, H, W)
            batch_size = input.shape[0]

        output = np.zeros((batch_size, *self.output_shape))

        for b in range(batch_size):
            for i in range(self.depth):
                output[b, i] = self.bias_matrix[i, 0, 0]  # Broadcast bias
                for j in range(self.input_depth):
                    # Ensure inputs are 2D
                    input_slice = input[b, j]
                    if input_slice.ndim == 1:  # Handle case where channel dimension was squeezed
                        input_slice = input_slice.reshape(self.input_height, self.input_width)
                    kernel_slice = self.kernel_matrix[i, j]

                    output[b, i] += signal.correlate2d(
                        input_slice,
                        kernel_slice,
                        mode='valid'
                    )

        # If we added a batch dimension for a single sample, remove it
        if batch_size == 1:
            output = output.squeeze(axis=0)

        return output

    def backward(self, grad_output, learning_rate):
        np.clip(grad_output, -1e5, 1e5, out=grad_output)

        batch_size = self.input.shape[0]
        kernels_gradient = np.zeros_like(self.kernel_matrix)
        input_gradient = np.zeros_like(self.input)

        # Calculate bias gradient
        bias_gradient = np.sum(grad_output, axis=(0, 2, 3)).reshape(-1, 1, 1)

        for n in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    # Ensure inputs are 2D
                    input_slice = self.input[n, j]
                    if input_slice.ndim == 1:
                        input_slice = input_slice.reshape(self.input_height, self.input_width)
                    grad_slice = grad_output[n, i]

                    if grad_slice.ndim == 1:
                        print("grad_slice 1 dim, reshaping")
                        grad_slice = grad_slice.reshape(self.output_shape[1], self.output_shape[2])

                    kernels_gradient[i, j] += signal.correlate2d(
                        input_slice,
                        grad_slice,
                        mode='valid'
                    )
                    np.clip(kernels_gradient, -1e3, 1e3, out=kernels_gradient)


                    input_grad = signal.convolve2d(
                        grad_slice,
                        self.kernel_matrix[i, j],
                        mode='full'
                    )
                    np.clip(input_grad, -1e3, 1e3, out=input_grad)

                    # Handle potential shape mismatches
                    h, w = self.input_height, self.input_width
                    input_gradient[n, j] += input_grad[:h, :w]

        self.kernel_matrix -= learning_rate * kernels_gradient
        self.bias_matrix -= learning_rate * bias_gradient

        return input_gradient








"""import numpy as np
from scipy import signal
from network_code.layer import Layer


class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_shape = (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernel_shape = (depth, self.input_depth, kernel_size, kernel_size)
        scale = np.sqrt(2 / (self.input_depth * kernel_size * kernel_size))
        self.kernel_matrix = np.random.randn(*self.kernel_shape) * scale
        self.bias_matrix = np.zeros((depth, 1, 1))  # Single bias per filter
        self.input = None

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        # print(input.shape)
        # print(f"input 0 : {input[0].shape}")
        # print(f"input 0, 0 : {input[0, 0].shape}")

        output = np.zeros((batch_size, *self.output_shape))


        for b in range(batch_size):
            for i in range(self.depth):
                output[b, i] = self.bias_matrix[i, 0, 0]  # Broadcast bias
                for j in range(self.input_depth):
                    print(f"input shape pre squeeze : {input.shape}")

                    # Ensure inputs are 2D
                    input_slice = np.squeeze(input[b, j])
                    # print(f"input_slice shape: {input_slice.shape}")
                    kernel_slice = np.squeeze(self.kernel_matrix[i, j])
                    print(f"input slice shape : {input_slice.shape}")

                    if input_slice.ndim != 2 or kernel_slice.ndim != 2:
                        raise ValueError(
                            f"Convolution requires 2D inputs. Got shapes: "
                            f"input {input_slice.shape}, kernel {kernel_slice.shape} b, j: {b,j}"
                        )

                    output[b, i] += signal.correlate2d(
                        input_slice,
                        kernel_slice,
                        mode='valid'
                    )
        return output

    def backward(self, grad_output, learning_rate):
        batch_size = self.input.shape[0]
        kernels_gradient = np.zeros_like(self.kernel_matrix)
        input_gradient = np.zeros_like(self.input)

        # Calculate bias gradient
        bias_gradient = np.sum(grad_output, axis=(0, 2, 3)).reshape(-1, 1, 1)

        for n in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    # Ensure inputs are 2D
                    input_slice = np.squeeze(self.input[n, j])
                    grad_slice = np.squeeze(grad_output[n, i])

                    kernels_gradient[i, j] += signal.correlate2d(
                        input_slice,
                        grad_slice,
                        mode='valid'
                    )

                    input_grad = signal.convolve2d(
                        grad_slice,
                        np.squeeze(self.kernel_matrix[i, j]),
                        mode='full'
                    )
                    # Handle potential shape mismatches
                    h, w = self.input_height, self.input_width
                    input_gradient[n, j] += input_grad[:h, :w]

        self.kernel_matrix -= learning_rate * kernels_gradient
        self.bias_matrix -= learning_rate * bias_gradient

        return input_gradient"""