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
