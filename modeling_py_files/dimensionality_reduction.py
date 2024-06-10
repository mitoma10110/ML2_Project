import matplotlib.pyplot as plt
import umap


def apply_umap(df, cluster_column='cluster_kmeans'):
    '''
    Function to visualize clusters through UMAP

    Inputs:
    - df: DataFrame containing the data.
    - cluster_column: Name of the column containing the cluster assignments.

    Outputs:
    - UMAP visualization of the clusters
    '''
    dataset = df.copy()

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, n_components=2, random_state=0)

    # Step 2: Fit and transform the data
    umap_embeddings = umap_model.fit_transform(dataset)

    # Step 3: Add UMAP embeddings to the DataFrame
    '''Assuming modeldf_purchase is a DataFrame'''
    dataset['umap_1'] = umap_embeddings[:, 0]
    dataset['umap_2'] = umap_embeddings[:, 1]
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(dataset['umap_1'], dataset['umap_2'], 
                        c=dataset[cluster_column], cmap='Spectral', s=10)

    '''Adding labels to the clusters'''
    for cluster in set(dataset[cluster_column]):
        plt.scatter([], [], c=scatter.cmap(scatter.norm(cluster)), label=f'Cluster {cluster}')

    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP Projection')
    plt.colorbar(scatter, label='Cluster')
    plt.show()



