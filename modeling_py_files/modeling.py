from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.metrics import confusion_matrix, silhouette_score
import numpy as np


def create_dispersion_list(data, k_limit=20):
    """
    Calculate the dispersion for different values of k in K-means clustering.

    Inputs:
    data (array-like): The input data to be clustered.
    k_limit (int, optional): The maximum number of clusters to consider. Defaults to 20.

    Outputs:
    list: A list of dispersion values for each value of k.

    """
    dispersion = []
    for k in range(1, k_limit):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dispersion.append(kmeans.inertia_)
    return dispersion

def calculate_silhouette_scores(data, k_values, random_state=42):
    """
    Calculate the silhouette scores for different values of k in KMeans clustering.

    Inputs:
    - data: The input data for clustering.
    - k_values: A list of integers representing the number of clusters to try.
    - random_state: An integer representing the random seed for reproducibility.

    Outputs:
    - silhouette_scores: A dictionary where the keys are the values of k and the values are the corresponding silhouette scores.
    """
    silhouette_scores = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores[k] = round(silhouette_avg, 3)

    return silhouette_scores

def print_dbscan_info(labels):
    """
    Prints information about the clusters and noise points obtained from DBSCAN algorithm.

    Inputs:
    - labels (array-like): The cluster labels assigned by DBSCAN algorithm.

    Outputs:
    - clusters (dict): A dictionary containing the cluster labels as keys and the number of points in each cluster as values.
    - noise_points (int): The number of noise points (points not assigned to any cluster).
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_info = dict(zip(unique_labels, counts))

    noise_points = cluster_info.get(-1, 0)
    clusters = {k: v for k, v in cluster_info.items() if k != -1}

    return clusters, noise_points

def create_agg_clusters(data, linkage='ward', distance_threshold=None, n_clusters=None):
    """
    Create agglomerative clusters using the given data.

    Inputs:
    - data: The input data for clustering.
    - linkage: The linkage criterion to use for clustering. Default is 'ward'.
    - distance_threshold: The threshold to use for clustering. Default is None.
    - n_clusters: The number of clusters to form. Default is None.

    Outputs:
    - agg_clust: The fitted AgglomerativeClustering model.

    """
    model = AgglomerativeClustering(linkage=linkage, distance_threshold=distance_threshold, n_clusters=n_clusters)
    agg_clust = model.fit(data)
    return agg_clust

def batch_meanshift(X, bandwidth, batch_size=10000):
    """
    Perform batch mean shift clustering on the input data.

    Inputs:
    - X (array-like): The input data to be clustered.
    - bandwidth (float): The bandwidth parameter for mean shift clustering.
    - batch_size (int, optional): The size of each batch. Defaults to 10000.

    Outputs:
    - labels (ndarray): The cluster labels assigned to each data point.

    """
    labels = np.empty(0)
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        mean_shift = MeanShift(bandwidth=bandwidth)
        mean_shift.fit(X_batch)
        labels = np.concatenate([labels, mean_shift.labels_])
    return labels

def allocate_clusters_kmeans(df, data_preprocessed, n_clusters=8, random_state=0):
    """
    Assigns cluster labels to the data points using K-means clustering algorithm.

    Inputs:
    - df (pandas.DataFrame): The DataFrame to which the cluster labels will be added.
    - data_preprocessed (numpy.ndarray): The preprocessed data used for clustering.
    - n_clusters (int): The number of clusters to create (default: 8).
    - random_state (int): The random seed for reproducibility (default: 0).

    Outputs:
    None
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data_preprocessed)
    df['cluster_kmeans'] = model.predict(data_preprocessed)


def allocate_clusters_hierarchical(df, data_preprocessed, distance_threshold=None, n_clusters=None, linkage='ward'):
    """
    Assigns cluster labels to the data points using hierarchical clustering.

    Inputs:
    - df: DataFrame
        The DataFrame to which the cluster labels will be added.
    - data_preprocessed: array-like
        The preprocessed data used for clustering.
    - distance_threshold: float, optional
        The linkage distance threshold at which to cut the dendrogram and form flat clusters.
        If not specified, the dendrogram will not be cut and the number of clusters will be determined by `n_clusters`.
    - n_clusters: int, optional
        The number of clusters to form if `distance_threshold` is not specified.
        If not specified, the number of clusters will be determined by `distance_threshold`.
    - linkage: str, optional
        The linkage criterion to use for clustering. Default is 'ward'.

    Outputs:
    None
    """
    model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters, linkage=linkage).fit(data_preprocessed)
    df['cluster_hierarchical'] = model.fit_predict(data_preprocessed) - 1 # since hierarchical is the only one that starts at 1


def allocate_clusters_dbscan(df, data_preprocessed, eps=0.5, min_samples=5):
    """
    Allocates clusters to the given dataframe using the DBSCAN algorithm.

    Inputs:
    - df (pandas.DataFrame): The dataframe to allocate clusters to.
    - data_preprocessed (numpy.ndarray): The preprocessed data used for clustering.
    - eps (float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood. Defaults to 0.5.
    - min_samples (int, optional): The minimum number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.

    Outputs:
    None
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster_dbscan'] = model.fit_predict(data_preprocessed)


def allocate_clusters_meanshift(df, data_preprocessed, bandwidth):
    """
    Allocates clusters to data points using the MeanShift algorithm.

    Inputs:
    - df (pandas.DataFrame): The DataFrame to which the cluster labels will be added.
    - data_preprocessed (numpy.ndarray): The preprocessed data on which the clustering will be performed.
    - bandwidth (float): The bandwidth parameter for the MeanShift algorithm.

    Outputs:
    None
    """
    model = MeanShift(bandwidth=bandwidth)
    df['cluster_meanshift'] = model.fit_predict(data_preprocessed)