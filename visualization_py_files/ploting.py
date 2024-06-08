import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster.hierarchy import dendrogram, linkage
from shapely.geometry import Point
import geopandas as gpd


def plot_population(data):
    # Assuming cust_info contains latitude and longitude columns for Portugal
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry)   

    # Load low-resolution world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Filter to include only Portugal
    portugal = world[world['name'] == 'Portugal']

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot map of Portugal
    portugal.plot(ax=ax, color='lightblue') 

    # Plot all points in Portugal
    gdf.plot(ax=ax, marker='o', color='red', markersize=15)

    # Set plot title and labels
    ax.set_title('Location of Customers in Portugal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_ylim(38, 40)
    ax.set_xlim(-9.8, -8)
    # Show plot
    plt.show()

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

def plot_cluster_description(modeldf_purchase):
    # Plotting cluster description using only the modelling variables
    cluster_means = modeldf_purchase.groupby('cluster_kmeans').mean()
    overall_mean = modeldf_purchase.mean()

    overall_mean_df = overall_mean.to_frame().T
    overall_mean_df.index = ['Overall Mean']
    plt.figure(figsize=(14, 8))

    # Plot each variable's mean for each cluster using clustered bar plots
    ax = cluster_means.T.plot(kind='bar', width=0.4, position=1.5, ax=None, figsize=(14, 8))

    # Plot the overall mean using a line plot on the same axes
    overall_mean_df.T.plot(kind='line', marker='o', linestyle='--', color='black', linewidth=2, ax=ax, label='Overall Mean')

    plt.title('Mean of Each Variable by Cluster with Overall Mean')
    plt.xlabel('Variables')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def plot_cluster_sizes(modeldf_purchase):
    modeldf_purchase.groupby(['cluster_kmeans']).size().plot(kind='bar')
    plt.show()