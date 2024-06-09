import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap


def visualize_dimensionality_reduction(transformation, targets):
    '''
    Author: Yehor Malakov
    20221691@novaims.unl.pt
    '''
    # Ensure that the targets are numpy array of integers
    targets = np.array(targets).astype(int)
    labels = np.unique(targets)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(transformation[:, 0], transformation[:, 1],
                          c=targets, cmap=plt.cm.tab20, s=10, alpha=0.7)
    
    # Create a legend with the class labels and colors
    handles = []
    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
    for label in labels:
        color = cmap(norm(label))
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(label),
                                  markerfacecolor=color, markersize=10))

    plt.legend(handles=handles, title='Classes', loc='best', ncol=1, fontsize='small')
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Dimensionality Reduction Visualization')
    plt.grid(True)
    plt.show()

# Example usage:
# Assuming you have a transformation result (e.g., from t-SNE or PCA) and corresponding targets
# transformation = np.random.rand(100, 2)  # example transformation data
# targets = np.random.randint(0, 20, size=100)  # example target labels
# visualize_dimensionality_reduction(transformation, targets)


def apply_tsne(dataset):
    '''
    Function to visualize clusters through T-SNE
    '''
    X_sample = dataset.iloc[:,1:].sample(20000)
    y_sample = dataset.iloc[:,0].loc[X_sample.index]

    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    # fit the data to the t-SNE model and transform it
    X_tsne = tsne_model.fit_transform(X_sample)

    visualize_dimensionality_reduction(X_tsne, y_sample)

    check_tsne_variables = pd.concat(
        [X_sample,
        pd.DataFrame(
            X_tsne,
            index=X_sample.index,
            columns=['tsne1','tsne2'])],
        axis=1)

    correlation_tsne = check_tsne_variables.corr()
    correlation_tsne.loc['tsne1',:].sort_values()

    # Check Influence
    mean_pixels = check_tsne_variables[['pixel445', 'pixel446', 'pixel447']].mean(axis=1)

    # create a scatter plot of the t-SNE output
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.array(mean_pixels))
    plt.colorbar(sc, label='Mean Pixel Values')
    plt.show()

    return correlation_tsne

def apply_umap(dataset):
    '''
    Function to visualize clusters through UMAP
    '''
    X_sample = dataset.iloc[:,1:].sample(20000)
    y_sample = dataset.iloc[:,0].loc[X_sample.index]

    # There's a but on this UMAP implementation and we also need to set numpy seed
    umap_object = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    # By some weird reason, random indexes don't work in this UMAP library
    X_sample = X_sample.sort_index()

    umap_embedding = umap_object.fit_transform(X_sample)

    # create a scatter plot of the UMAP output
    visualize_dimensionality_reduction(umap_embedding, y_sample.sort_index())