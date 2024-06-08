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
    # create a scatter plot of the t-SNE output
    plt.scatter(transformation[:, 0], transformation[:, 1],
                c=np.array(targets).astype(int), cmap=plt.cm.tab20)

    labels = np.unique(targets)

    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=min(np.array(labels).astype(int)), vmax=max(np.array(labels).astype(int)))
    rgba_values = cmap(norm(labels))

    # create a legend with the class labels and colors
    handles = [plt.scatter([], [], c=rgba, label=label) for rgba, label in zip(rgba_values, labels)]
    plt.legend(handles=handles, title='Classes')

    plt.show()

def apply_tsne(dataset):
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
    X_sample = dataset.iloc[:,1:].sample(20000)
    y_sample = dataset.iloc[:,0].loc[X_sample.index]

    # There's a but on this UMAP implementation and we also need to set numpy seed
    umap_object = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    # By some weird reason, random indexes don't work in this UMAP library
    X_sample = X_sample.sort_index()

    umap_embedding = umap_object.fit_transform(X_sample)

    # create a scatter plot of the UMAP output
    visualize_dimensionality_reduction(umap_embedding, y_sample.sort_index())