from sklearn.datasets import load_iris
import pandas as pd
import random
import numpy as np

seed = 456

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#iris_df = iris_df.sample(frac=1, random_state=seed)
iris_bk = iris_df.copy()
y_target = iris.target
#iris_df = iris_df.drop(columns=["species"])

print(iris_df.head())
print(y_target)


def convergence_criteria(centroids,old_centroids,k):
    sum = 0
    for i in range(k):
        sum += np.sqrt(np.sum((old_centroids[i] - centroids[i]) ** 2))

    if sum <= 1e-6:
        return True
    else:
        return False

def search_min_index(centroids,xj,k):
    min_index = 0
    current_lower_dist =  np.sqrt(np.sum((xj - centroids[0]) ** 2))
    for i in range(k):
        dist = np.sqrt(np.sum((xj - centroids[i]) ** 2))
        if dist < current_lower_dist:
            current_lower_dist = dist
            min_index = i
    return min_index

def LLoyds(centroids,iris_df,m,k,d,tol=1e-6):

    old_centroids = np.zeros_like(centroids)
    positions = [0 for i in range(m)]
    size = [0 for i in range(k)]
    sum = np.zeros((k,d))
    ite = 0
    while convergence_criteria(centroids,old_centroids,k) is False:
        for j in range(m):
            xj = iris_df.iloc[j].to_numpy()
            min_index = search_min_index(centroids,xj,k)
            if positions[j] != min_index:
                if size[positions[j]] > 0: size[positions[j]] -= 1
                if not (np.all(sum[positions[j]]) == 0): sum[positions[j]] -= iris_df.iloc[j]

                size[positions[min_index]] += 1
                sum[positions[min_index]] += iris_df.iloc[j]
                positions[j] = min_index
        for i in range(k):
            old_centroids[i] = centroids[i]
            if size[i] != 0 : centroids[i] = sum[i] / size[i]
        ite += 1

    print(ite)
    return positions


d = len(iris_df.columns) #number of features
k = 3 #number of clusters
m = iris_df.shape[0] #number of samples
centroids = np.zeros((d, k))

for i in range(d): #initialization of centroids
    max_feat_d = iris_df.iloc[:, i].max()
    min_feat_d = iris_df.iloc[:, i].min()
    print(max_feat_d, min_feat_d)
    for j in range(k):
        centroids[i][j] = (random.uniform(min_feat_d, max_feat_d)).round(3)

centroids = centroids.T
print(centroids)
cluster = LLoyds(centroids,iris_df,m,k,d)
print(cluster)
err = 0
#for i in range(m):
    #if cluster[i] == iris_df.iloc[i]['species']:
   #     err += 1

print(err/m)
'''Let' start with a fixed number of cluster. Let's choose k = 3'''