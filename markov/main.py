from model import HMM
from preprocess_image import encode_image_to_sequence, train_kmeans_on_strips, get_all_strips

IMG_SIZE = (28, 28)    # Resize all images to this
NUM_STRIPS = 16        # Sequence length
DATA_DIR = "data"
DATA_TEST = "test"

# 1. Gather strips from all images
strips, image_paths, labels = get_all_strips(DATA_DIR)

# 2. Train clusterer
kmeans = train_kmeans_on_strips(strips, n_clusters=16)

# 3. Encode all images into discrete sequences
sequences_by_label = {}

for path, label in zip(image_paths, labels):
    seq = encode_image_to_sequence(path, kmeans)
    if label not in sequences_by_label:
        sequences_by_label[label] = []
    sequences_by_label[label].append(seq)
print("Sequences for label 'A':", sequences_by_label["A"])



model = HMM(n_states=4, n_observations=5)
model.baum_welch(sequences_by_label["A"], max_iter=20)


# Suppose you encode images into discrete sequences (e.g. via k-means on strips)

test_strips, image_paths2, labels = get_all_strips(DATA_TEST)
kmeans2 = train_kmeans_on_strips(test_strips, n_clusters=16)

print(f"image path2 is {image_paths2}")

image_paths2 = str(image_paths2)

test_seq = encode_image_to_sequence(image_paths2, kmeans2)
print(test_seq)
log_prob = model.score(test_seq)
print("Log-likelihood:", log_prob)
