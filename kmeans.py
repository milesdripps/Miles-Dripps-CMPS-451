# Miles Dripps CMPS 451 K means algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs


# Generate synthetic data
def generate_data(n_samples=1000, n_features=2, centers=3, random_state=42):

    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    return X, y_true


# Scale  data
def scale_data(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Perform KMeans clustering
def kmeans_clustering(X, n_clusters=3, random_state=42):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    y_kmeans = kmeans.fit_predict(X)
    return kmeans, y_kmeans


# Plot the results
def plot_clusters(X, y_kmeans, kmeans, title="KMeans Clustering"):

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
    plt.title(title)
    plt.show()


# Main function to run KMeans
def main():
    X, y_true = generate_data()

    X_scaled = scale_data(X)

    kmeans, y_kmeans = kmeans_clustering(X_scaled)

    plot_clusters(X_scaled, y_kmeans, kmeans)


if __name__ == "__main__":
    main()
