import pickle
import numpy as np
import time

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data, batch_size=32):
        if input_data.ndim == 3:
            input_data = input_data[np.newaxis, ...]  # Add batch dimension

        num_samples = len(input_data)
        results = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = input_data[start:end]
            output = batch

            print(f"\nBatch {start // batch_size} input shape: {output.shape}")

            for j, layer in enumerate(self.layers):
                output = layer.forward(output)

            if output.ndim == 3 and output.shape[-1] == 1:
                output = output.squeeze(-1)



            results.append(output)

        return np.concatenate(results, axis=0)

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size=32):
        start_time = time.time()
        k: int = 0
        total_batches = int(np.ceil(len(x_train) / batch_size))  # Calculate total batches per epoch (rounding up)

        for epoch in range(epochs):
            error = 0
            for i in range(0, len(x_train), batch_size):
                k += 1
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)
                    assert not np.any(np.isnan(output)), "NaN detected in forward pass"

                # Remove extra dimension if present before loss calculation
                if output.ndim == 3:
                    output = output.squeeze(axis=-1)

                #compute loss
                error += np.sum(self.loss(y_batch, output))

                # Backpropagation
                grad = self.loss_prime(y_batch, output)
                # Add extra dimension if needed
                if grad.ndim == 2:
                    grad = grad[..., np.newaxis]

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
                    assert not np.any(np.isnan(grad)), "NaN detected in backward pass"

                if k % 1 == 0:  # Print every 10 batches (you can adjust this)
                    elapsed_time = time.time() - start_time
                    percentage_complete = (k / (
                                epochs * total_batches)) * 100  # Percentage based on total batches across epochs
                    fraction_complete = f"{k}/{epochs * total_batches}"
                    print(
                        f"Epoch {epoch + 1}/{epochs} - Batch {fraction_complete} - {percentage_complete:.2f}% done - Time elapsed: {elapsed_time:.2f}s")

            error /= len(x_train)
            print(f"Epoch {epoch + 1}/{epochs} - Error: {error:.4f}")

    def save(self, path="model.pkl"):
        # Create a serializable version
        import copy
        temp = copy.copy(self)

        # Remove problematic attributes
        if hasattr(temp, 'loss') and callable(temp.loss):
            temp._loss_name = temp.loss.__name__
            temp.loss = None
        if hasattr(temp, 'loss_prime') and callable(temp.loss_prime):
            temp._loss_prime_name = temp.loss_prime.__name__
            temp.loss_prime = None

        with open(path, "wb") as f:
            pickle.dump(temp, f)

    @staticmethod
    def load(path="model.pkl"):
        with open(path, "rb") as f:
            net = pickle.load(f)

        # Restore functions if needed
        if hasattr(net, '_loss_name'):
            net.loss = globals()[net._loss_name]
        if hasattr(net, '_loss_prime_name'):
            net.loss_prime = globals()[net._loss_prime_name]

        return net