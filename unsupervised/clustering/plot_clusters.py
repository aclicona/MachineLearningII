import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
def plot_clusters(data, labels):
    n_clusters = len(np.unique(labels))
    # Create a subplot with 1 row and 2 columns
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(6, 6)



    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax1.scatter(
        data[:, 0], data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    random_centers = np.array([np.random.choice(np.where(labels == label)[0], 1)[0] for label in np.unique(labels)])
    # Labeling the clusters
    centers = data[random_centers]
    # Draw white circles at cluster centers
    ax1.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax1.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax1.set_title("The visualization of the clustered data.")
    ax1.set_xlabel("Feature space for the 1st feature")
    ax1.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


    plt.show()

