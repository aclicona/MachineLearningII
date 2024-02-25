
import numpy as np



class KMedoids:
    def __init__(self, data, n_clusters):
        self.n_clusters = n_clusters
        self.data = data

    @staticmethod
    def calculate_medoids(data, axis=1):
        medoids = np.median(data, axis=axis)
        return medoids

    def initialize_medoids(self):
        # Initialize the medoids
        ramdom_indices = np.random.randint(0, self.n_clusters, len(self.data))
        data_divided = [self.data[ramdom_indices == cluster_index] for cluster_index in range(self.n_clusters)]
        return np.array([np.median(data, axis=0) for data in data_divided])

    def assign_clusters(self, medoids):
        # Assign each data point to the closest medoid
        distances = np.sqrt(((self.data - medoids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_assignment = np.argmin(distances, axis=0)
        return cluster_assignment

    def update_medoids(self, cluster_assignment):
        # Update medoids based on cluster assignments
        new_medoids = np.array(
            [self.calculate_medoids(self.data[cluster_assignment == i], axis=0) for i in range(self.n_clusters)])
        return new_medoids

    def fit(self, max_iterations=100):
        # Fit KMedoids to the data
        medoids = self.initialize_medoids()
        for _ in range(max_iterations):
            cluster_assignment = self.assign_clusters(medoids)
            # new_medoids = self.update_medoids(cluster_assignment)
            new_medoids = self.update_medoids(cluster_assignment)
            if np.allclose(medoids, new_medoids):
                break
            medoids = new_medoids
        return medoids, cluster_assignment
