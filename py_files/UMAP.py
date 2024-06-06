import umap
from t_SNE import visualize_dimensionality_reduction

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