from sklearn.cluster import KMeans, AgglomerativeClustering

def create_dispersion_list(data, k_limit=20):
    dispersion = []
    for k in range(1, k_limit):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dispersion.append(kmeans.inertia_)
    return dispersion

def create_agg_clusters(data, linkage='ward', distance_threshold=0, n_clusters=None):
    agg_clust = AgglomerativeClustering(
        linkage=linkage, distance_threshold=distance_threshold, n_clusters=n_clusters
        ).fit(data)
    return agg_clust

def allocate_clusters(df, data_preprocessed, n_clusters, cluster_type=['KMeans', 'AgglomerativeClustering'], random_state=0, linkage='ward'):
    if cluster_type=='KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data_preprocessed)
        df['cluster_kmeans'] = model.predict(data_preprocessed)
    elif cluster_type=='AgglomerativeClustering':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(data_preprocessed)
        df['cluster_hierarchical'] = model.fit_predict(data_preprocessed)