import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans

IMG_SIZE = (32, 32)    # Resize all images to this
NUM_STRIPS = 16        # Sequence length
DATA_DIR = "data"      # Each subfolder is a letter class

def get_all_strips(data_dir):
    strips = []
    labels = []
    image_paths = []

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.endswith('.png'):
                continue
            path = os.path.join(class_dir, fname)
            img = imread(path, as_gray=True)
            img = 1.0 - resize(img, IMG_SIZE)  # Black bg, white fg

            strip_width = IMG_SIZE[1] // NUM_STRIPS
            for i in range(NUM_STRIPS):
                strip = img[:, i*strip_width:(i+1)*strip_width]
                strips.append(strip.flatten())
            image_paths.append(path)
            labels.append(label)

    return np.array(strips), image_paths, labels

def train_kmeans_on_strips(strips, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(strips)
    return kmeans

def encode_image_to_sequence(img_path, kmeans):
    img = imread(img_path, as_gray=True)
    img = 1.0 - resize(img, IMG_SIZE)
    strip_width = IMG_SIZE[1] // NUM_STRIPS
    sequence = []

    for i in range(NUM_STRIPS):
        strip = img[:, i*strip_width:(i+1)*strip_width].flatten()
        symbol = kmeans.predict([strip])[0]
        sequence.append(symbol)

    return sequence  # List[int]
