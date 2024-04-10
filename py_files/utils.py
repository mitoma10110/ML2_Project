import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster.hierarchy import dendrogram, linkage
import math
from PIL import Image
import urllib
from sklearn.cluster import KMeans, AgglomerativeClustering

def create_dispersion_list(data, k_limit=20):
    dispersion = []
    for k in range(1, k_limit):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dispersion.append(kmeans.inertia_)
    return dispersion

# plot funtions for k analysis
def plot_elbow_graph(dispersion):
    '''
    Function to plot elbow graphs given a dispersion list
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 20), dispersion, marker='o')
    plt.xticks(range(1, 21, 1))
    plt.xlabel('Number of clusters')
    plt.ylabel('Dispersion (inertia)')
    plt.show()

def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)