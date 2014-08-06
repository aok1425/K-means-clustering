import grab_values
from sklearn import cluster, datasets

# doing K-means clustering
values = grab_values.values
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(values)

# doing PCA
from sklearn import decomposition

pca = decomposition.PCA()
pca.fit(values)
print pca.explained_variance_

# [ 0.75073447  0.587651    0.0874056   0.02311994]

pca.n_components = 2
X_reduced = pca.fit_transform(values)
X_reduced.shape