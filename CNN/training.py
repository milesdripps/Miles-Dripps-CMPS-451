import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from network_code.neural_network import NeuralNetwork
from network_code.convolution import Convolution
from network_code.reshape import Reshape
from network_code.dense import Dense
from network_code.activation import Activation
from network_code.pooling import MaxPooling
from network_code.layer_norm import BatchNormalization
from network_code.activation_fns import relu, softmax, relu_prime, softmax_derivative
from network_code.losses import categorical_crossentropy, categorical_crossentropy_gradient

# Configuration
DATASET_PATH = "../datasets/plant"
IMAGE_SIZE = (256, 256)
EPOCHS = 1
LEARNING_RATE = 0.01

def evaluate_model(net, x_test, y_test, class_names):
    print(f"Evaluating model against test dataset")
    predictions = net.predict(x_test, batch_size=32)
    accuracy = np.mean([np.argmax(y_test[i]) == np.argmax(predictions[i])
                        for i in range(len(y_test))])
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print("------------------")
    return accuracy


def load_data(path, image_size, test_size=0.2, random_state=42):
    X, y = [], []
    class_names = sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')
    ])
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    for class_name in tqdm(class_names, desc="Loading classes"):
        class_dir = os.path.join(path, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]

        for file in tqdm(image_files, desc=class_name, leave=False):
            try:
                img = Image.open(os.path.join(class_dir, file)).convert('RGB')
                img = img.resize(image_size)
                img_array = np.asarray(img, dtype=np.float32) / 255.0
                img_array = img_array.transpose(2, 0, 1)  # CHW
                X.append(img_array)
                label = np.zeros(len(class_names))
                label[class_to_index[class_name]] = 1
                y.append(label)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nLoaded {len(X)} images from {len(class_names)} classes")
    print(f"Image shape: {X.shape[1:]}, Label shape: {y.shape[1:]}")
    print("Splitting dataset : test/train")
    return train_test_split(X, y, test_size=test_size, random_state=random_state), class_names

def train():
    (x_train, x_test, y_train, y_test), class_names = load_data(DATASET_PATH, IMAGE_SIZE)
    print("Initializing model")

    input_depth = 3
    height, width = IMAGE_SIZE

    net = NeuralNetwork()

    # First Conv Block
    net.add(Convolution(input_shape=(input_depth, height, width), kernel_size=3, depth=8))
    net.add(BatchNormalization(8))
    net.add(Activation(relu, relu_prime))
    net.add(MaxPooling(size=2))  # Reduces spatial dimensions by half

    # Calculate dimensions after first block
    new_height = (height - 3 + 1) // 2  # (256-3+1)/2 = 127
    new_width = (width - 3 + 1) // 2  # same

    # Second Conv Block
    net.add(Convolution(input_shape=(8, new_height, new_width), kernel_size=5, depth=16))
    net.add(BatchNormalization(16))
    net.add(Activation(relu, relu_prime))
    net.add(MaxPooling(size=2))  # Reduces spatial dimensions by half again

    # Calculate dimensions after second block
    new_height = (new_height - 3 + 1) // 2  # (127-3+1)/2 = 62
    new_width = (new_width - 3 + 1) // 2  # same

    # Flatten layer
    flat_size = 16 * new_height * new_width  # 16*62*62 = 61,504
    net.add(Reshape((16, new_height, new_width), (flat_size, 1)))

    # Dense layers
    net.add(Dense(flat_size, 128))
    net.add(Activation(relu, relu_prime))
    net.add(Dense(128, len(class_names)))
    net.add(Activation(softmax, softmax_derivative))

    net.use(categorical_crossentropy, categorical_crossentropy_gradient)

    net.fit(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=32)

    net.save("cnn_multiclass_model.pkl")
    print("Model saved to cnn_multiclass_model.pkl")

    evaluate_model(net, x_test, y_test, class_names)


if __name__ == "__main__":
    train()