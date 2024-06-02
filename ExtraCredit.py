import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

def assign_clusters(dataset, subset_size):
    '''
    Function that assigns clusters to data points in a dataset using Agglomerative Clustering,
    by fitting th emodel with only part of the dataset, then gives clusters to all the other points
    by calculating euclidean distances to the centroids

    Inputs:
        - dataset (Dataframe): the dataset containing the data points
        - subset_size (int): the size of the subset for fitting the model

    Output:
        - dataframe: dataset with the cluster column
    '''

    # fitting the model
    fit_set = dataset[:subset_size]
    model = AgglomerativeClustering()
    model.fit(fit_set)

    # cluster centroids coordinates
    centroids = np.array([fit_set[model.labels_ == label].mean(axis=0) for label in range(model.n_clusters_)])

    # matrix of euclidean distances between points and centroids
    distances = cdist(dataset.values, centroids, 'euclidean')

    # get the closest centroid's column index
    closest_centroid_index = distances.argmin(axis=1)

    # mapping from cluster index to letter
    cluster_mapping = {idx: chr(65 + idx) for idx in range(model.n_clusters_)}
    closest_cluster_letters = np.array([cluster_mapping[idx] for idx in closest_centroid_index])

    dataset['cluster'] = closest_cluster_letters

    return dataset


# Testing --------------------------------------------------------------
# Note: I asked chat for this, no idea "make_blobs" was a thing
# We should probably learn this for testing clustering algorithms, seems useful

from sklearn.datasets import make_blobs

# Generate synthetic dataset
n_samples = 1000
n_features = 4
centers = 3  # Number of clusters
random_state = 42

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)

# Example usage:
subset_size = 100
predicted_labels = assign_clusters(pd.DataFrame(X), subset_size)
print(predicted_labels)
