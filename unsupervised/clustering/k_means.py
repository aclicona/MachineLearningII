
import numpy as np


class KMeans:
    def __init__(self, data, n_clusters):
        self.n_clusters = n_clusters
        self.data = data

    def initialize_centroids(self):
        # Randomly initialize centroids
        np.random.seed(300)
        centroids_indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
        centroids = self.data[centroids_indices]
        return centroids

    def assign_clusters(self, centroids):
        # Assign each data point to the closest centroid
        distances = np.sqrt(((self.data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_assignment = np.argmin(distances, axis=0)
        return cluster_assignment

    def update_centroids(self, cluster_assignment):
        # Update centroids based on cluster assignments
        new_centroids = np.array([self.data[cluster_assignment == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, max_iterations=200):
        # Fit KMeans to the data
        centroids = self.initialize_centroids()
        for _ in range(max_iterations):
            cluster_assignment = self.assign_clusters(centroids)
            new_centroids = self.update_centroids(cluster_assignment)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return centroids, cluster_assignment


