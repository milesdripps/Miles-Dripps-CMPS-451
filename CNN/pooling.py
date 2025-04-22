import numpy as np
from network_code.layer import Layer

class MaxPooling(Layer):
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, input):
        self.input = input
        batch_size, depth, height, width = input.shape
        out_height = (height - self.size) // self.stride + 1
        out_width = (width - self.size) // self.stride + 1
        output = np.zeros((batch_size, depth, out_height, out_width))

        for b in range(batch_size):
            for c in range(depth):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.size
                        w_end = w_start + self.size
                        output[b, c, i, j] = np.max(input[b, c, h_start:h_end, w_start:w_end])

        return output

    def backward(self, grad_output, learning_rate):
        np.clip(grad_output, -1e5, 1e5, out=grad_output)
        input_gradient = np.zeros_like(self.input)
        batch_size, depth, out_height, out_width = grad_output.shape

        for b in range(batch_size):
            for c in range(depth):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.size
                        w_end = w_start + self.size
                        window = self.input[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        for m in range(self.size):
                            for n in range(self.size):
                                if window[m, n] == max_val:
                                    input_gradient[b, c, h_start + m, w_start + n] += grad_output[b, c, i, j]
        return input_gradient
