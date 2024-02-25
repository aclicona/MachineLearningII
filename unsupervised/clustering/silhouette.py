import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Sillhouette:
    silhouette = None

    def __init__(self, data, clusters_assignment, centroids):
        self.data = data
        self.clusters_assignment = clusters_assignment
        self.centroids = centroids

    def mean_distance_to_nearest_neighboring_cluster(self, data_point, centroid_of_cluster):
        # Calculate the centroids other than its own
        mask = np.isin(self.centroids, centroid_of_cluster)
        # Use boolean indexing to filter elements
        other_cluster_centroids = self.centroids[mask]
        # Calculate the nearest centroid other than its own
        nearest_neighboring_centroid = self.nearest_neighboring_centroid(data_point, other_cluster_centroids)
        nearest_neighboring_cluster_data = self.data[np.where(self.centroids == nearest_neighboring_centroid)[0][0]]
        distances = np.linalg.norm(nearest_neighboring_cluster_data - data_point)
        return distances

    def nearest_neighboring_centroid(self, data_point, other_cluster_centroids):
        nearest_centroid = min(other_cluster_centroids, key=lambda x: np.linalg.norm(x - data_point))
        return nearest_centroid

    def mean_distance(self, cluster):
        """
        Compute the mean distance between each element of an array with the other elements of the same array.

        Parameters:
        - data: numpy array containing the data points

        Returns:
        - mean_distances: numpy array containing the mean distances for each element
        """
        num_points = len(cluster)
        mean_distances = np.zeros(num_points)

        for i in range(num_points):
            distances = np.linalg.norm(cluster - cluster[i], axis=1)  # Calculate Euclidean distance
            distances_without_self = np.delete(distances, i)  # Remove distance to itself
            mean_distances[i] = np.mean(distances_without_self)

        return mean_distances

    def calculate_silhouette(self):
        mean_cluster_distance, mean_distance_to_nearest_neighboring_cluster = np.ones(len(self.data)), np.ones(
            len(self.data))
        for cluster_index in range(len(self.centroids)):
            cluster_mask = self.clusters_assignment == cluster_index
            cluster_elements = self.data[cluster_mask]
            mean_cluster_distance[cluster_mask] = self.mean_distance(cluster_elements)
            mean_distance_to_nearest_neighboring_cluster[cluster_mask] = np.array(
                [self.mean_distance_to_nearest_neighboring_cluster(element, self.centroids[cluster_index]) for element
                 in
                 cluster_elements])
        max_distance = np.array([mean_cluster_distance, mean_distance_to_nearest_neighboring_cluster]).max(axis=0)
        self.silhouette = (mean_distance_to_nearest_neighboring_cluster - mean_cluster_distance) / max_distance
        return self.silhouette

    def plot_silhouette(self):
        n_clusters = len(self.centroids)
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(self.data) + (n_clusters + 1) * 10])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = self.silhouette.mean()
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample

        y_lower = 10
        for cluster_index in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = self.silhouette[self.clusters_assignment == cluster_index]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(cluster_index) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_index))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(self.clusters_assignment.astype(float) / n_clusters)
        ax2.scatter(
            self.data[:, 0], self.data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = self.centroids
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()
