import numpy as np

class HMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations

        # Random init
        self.A = np.full((self.N, self.N), 1 / self.N)
        self.B = np.full((self.N, self.M), 1 / self.M)
        self.pi = np.full(self.N, 1 / self.N)

    def forward(self, obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))

        alpha[0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            # Debugging: Check if the observation index is valid
            if obs_seq[t] >= self.M or obs_seq[t] < 0:
                print(f"Invalid observation index at t={t}: {obs_seq[t]}")
                return None  # or handle error gracefully

            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, obs_seq[t]]

        return alpha

    def backward(self, obs_seq):
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        beta[-1] = 1

        for t in reversed(range(T - 1)):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_seq[t + 1]] * beta[t + 1])

        return beta

    def baum_welch(self, sequences, max_iter=10):
        for iteration in range(max_iter):
            A_num = np.zeros_like(self.A)
            A_den = np.zeros(self.N)
            B_num = np.zeros_like(self.B)
            B_den = np.zeros(self.N)
            pi_new = np.zeros(self.N)

            for obs_seq in sequences:
                T = len(obs_seq)
                alpha = self.forward(obs_seq)
                beta = self.backward(obs_seq)

                xi = np.zeros((T - 1, self.N, self.N))
                for t in range(T - 1):
                    denom = np.sum(alpha[t] @ self.A * self.B[:, obs_seq[t+1]] * beta[t+1])
                    for i in range(self.N):
                        numer = alpha[t, i] * self.A[i, :] * self.B[:, obs_seq[t+1]] * beta[t+1]
                        xi[t, i, :] = numer / denom

                gamma = (alpha * beta)
                gamma /= gamma.sum(axis=1, keepdims=True)

                A_num += np.sum(xi, axis=0)
                A_den += np.sum(gamma[:-1], axis=0)

                for t in range(T):
                    B_num[:, obs_seq[t]] += gamma[t]
                B_den += np.sum(gamma, axis=0)

                pi_new += gamma[0]

            self.A = A_num / A_den[:, None]
            self.B = B_num / B_den[:, None]
            self.pi = pi_new / len(sequences)

    def score(self, obs_seq):
        alpha = self.forward(obs_seq)
        return np.log(np.sum(alpha[-1]))
