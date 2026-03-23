from sklearn.datasets import load_iris
import pandas as pd
import random
import numpy as np
import functions_clustering as fc

seed = 456

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y_target = iris.target

# shuffle mantenendo l'allineamento
shuffled_idx = np.random.permutation(len(iris_df))
iris_df = iris_df.iloc[shuffled_idx].reset_index(drop=True)
y_target = y_target[shuffled_idx]


print(iris_df.head())
print(y_target)

n_features = len(iris_df.columns) #number of features
n_clusters = 3 #number of clusters
m = iris_df.shape[0] #number of samples
centroids = np.zeros((n_clusters, n_features))

centroids = fc.initialize_centroids_kpp(n_clusters, iris_df)

fc.Lloyds(centroids, iris_df, m, n_clusters, n_features, tol=1e-6)