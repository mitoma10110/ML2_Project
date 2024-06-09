from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift


def create_dispersion_list(data, k_limit=20):
    dispersion = []
    for k in range(1, k_limit):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dispersion.append(kmeans.inertia_)
    return dispersion

def create_agg_clusters(data, linkage='ward', distance_threshold=None, n_clusters=None):
    model = AgglomerativeClustering(linkage=linkage, distance_threshold=distance_threshold, n_clusters=n_clusters)
    agg_clust = model.fit(data)
    return agg_clust


def allocate_clusters_kmeans(df, data_preprocessed, n_clusters=8, random_state=0):
    model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data_preprocessed)
    df['cluster_kmeans'] = model.predict(data_preprocessed)

def allocate_clusters_hierarchical(df, data_preprocessed, distance_threshold=None, n_clusters=None, linkage='ward'):
    model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters, linkage=linkage).fit(data_preprocessed)
    df['cluster_hierarchical'] = model.fit_predict(data_preprocessed)

def allocate_clusters_dbscan(df, data_preprocessed, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster_dbscan'] = model.fit_predict(data_preprocessed)

def allocate_clusters_meanshift(df, data_preprocessed, bandwidth):
    model = MeanShift(bandwidth=bandwidth)
    df['cluster_meanshift'] = model.fit_predict(data_preprocessed)