import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans


class HMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations

        # Random initialization with slight variation
        self.A = np.random.dirichlet(np.ones(self.N), size=self.N)
        self.B = np.random.dirichlet(np.ones(self.M), size=self.N)
        self.pi = np.random.dirichlet(np.ones(self.N))

    def forward(self, obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))

        # Initialization
        alpha[0] = self.pi * self.B[:, obs_seq[0]]
        alpha[0] /= np.sum(alpha[0])  # Normalization

        # Induction
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, obs_seq[t]]
            alpha[t] /= np.sum(alpha[t])  # Normalization

        return alpha

    def backward(self, obs_seq):
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        beta[-1] = 1.0

        for t in reversed(range(T - 1)):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_seq[t + 1]] * beta[t + 1])
            beta[t] /= np.sum(beta[t])  # Normalization

        return beta

    def baum_welch(self, sequences, max_iter=20, tolerance=1e-6):
        prev_log_prob = -np.inf

        for iteration in range(max_iter):
            A_num = np.zeros((self.N, self.N))
            A_den = np.zeros(self.N)
            B_num = np.zeros((self.N, self.M))
            B_den = np.zeros(self.N)
            pi_new = np.zeros(self.N)
            current_log_prob = 0

            for obs_seq in sequences:
                T = len(obs_seq)
                alpha = self.forward(obs_seq)
                beta = self.backward(obs_seq)

                # Calculate xi (transition probabilities between t and t+1)
                xi = np.zeros((T - 1, self.N, self.N))
                for t in range(T - 1):
                    xi[t] = (alpha[t].reshape(-1, 1) * self.A *
                             self.B[:, obs_seq[t + 1]].reshape(1, -1) *
                             beta[t + 1].reshape(1, -1))
                    xi[t] /= np.sum(xi[t])

                # Calculate gamma (state probabilities at each time)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)

                # Accumulate statistics
                A_num += np.sum(xi, axis=0)
                A_den += np.sum(gamma[:-1], axis=0)

                for t in range(T):
                    B_num[:, obs_seq[t]] += gamma[t]
                B_den += np.sum(gamma, axis=0)

                pi_new += gamma[0]
                current_log_prob += np.log(np.sum(alpha[-1]))

            # Update parameters
            self.A = A_num / A_den.reshape(-1, 1)
            self.B = B_num / B_den.reshape(-1, 1)
            self.pi = pi_new / len(sequences)

            # Check for convergence
            if iteration > 0 and abs(current_log_prob - prev_log_prob) < tolerance:
                break
            prev_log_prob = current_log_prob

    def score(self, obs_seq):
        alpha = self.forward(obs_seq)
        return np.log(np.sum(alpha[-1]))


def get_all_strips(data_dir, img_size=(32, 32), num_strips=16):
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
            try:
                img = imread(path, as_gray=True)
                img = 1.0 - resize(img, img_size)  # Invert: white on black

                strip_width = img_size[1] // num_strips
                for i in range(num_strips):
                    strip = img[:, i * strip_width:(i + 1) * strip_width]
                    strips.append(strip.flatten())

                image_paths.append(path)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return np.array(strips), image_paths, labels


def encode_image_to_sequence(img_path, kmeans, img_size=(32, 32), num_strips=16):
    try:
        img = imread(img_path, as_gray=True)
        img = 1.0 - resize(img, img_size)
        strip_width = img_size[1] // num_strips
        sequence = []

        for i in range(num_strips):
            strip = img[:, i * strip_width:(i + 1) * strip_width].flatten()
            symbol = kmeans.predict([strip])[0]
            sequence.append(symbol)

        return sequence
    except Exception as e:
        print(f"Error encoding {img_path}: {e}")
        return None


def main():
    # Parameters
    IMG_SIZE = (32, 32)
    NUM_STRIPS = 16
    N_CLUSTERS = 20  # Number of symbols for KMeans
    N_STATES = 5  # Number of hidden states

    # 1. Load and process training data
    print("Loading training data...")
    train_strips, train_paths, train_labels = get_all_strips("data", IMG_SIZE, NUM_STRIPS)

    # 2. Train KMeans clusterer on all training strips
    print("Training KMeans...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(train_strips)

    # 3. Encode training sequences
    print("Encoding training sequences...")
    sequences_by_label = {}

    for path, label in zip(train_paths, train_labels):
        seq = encode_image_to_sequence(path, kmeans, IMG_SIZE, NUM_STRIPS)
        if seq is None:
            continue
        if label not in sequences_by_label:
            sequences_by_label[label] = []
        sequences_by_label[label].append(seq)

    # 4. Train HMM for letter 'A'
    print("Training HMM for 'A'...")
    hmm_a = HMM(n_states=N_STATES, n_observations=N_CLUSTERS)
    hmm_a.baum_welch(sequences_by_label["A"], max_iter=20)

    # 5. Load and process test data
    print("Loading test data...")
    test_strips, test_paths, test_labels = get_all_strips("test", IMG_SIZE, NUM_STRIPS)

    # 6. Test the HMM
    print("\nTesting HMM on test images:")
    for path, label in zip(test_paths, test_labels):
        seq = encode_image_to_sequence(path, kmeans, IMG_SIZE, NUM_STRIPS)
        if seq is None:
            continue

        score = hmm_a.score(seq)
        print(f"Image: {os.path.basename(path)}, True Label: {label}, Score: {score:.2f}")


if __name__ == "__main__":
    main()