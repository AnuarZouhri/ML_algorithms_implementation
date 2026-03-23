import numpy as np

def convergence_criteria(centroids, old_centroids, n_clusters, tol=1e-6):
    sum = 0
    for i in range(n_clusters):
        sum += np.sqrt(np.sum((old_centroids[i] - centroids[i]) ** 2))

    if sum <= tol:
        return True
    else:
        return False


def search_min_index(centroids, xj, n_clusters):
    min_index = 0
    current_lower_dist =  np.sqrt(np.sum((xj - centroids[0]) ** 2))
    for i in range(n_clusters):
        dist = np.sqrt(np.sum((xj - centroids[i]) ** 2))
        if dist < current_lower_dist:
            current_lower_dist = dist
            min_index = i
    return min_index

def initialize_centroids(centroids, n_clusters, dataset):  #initialization of centroids
    indices = np.random.choice(dataset.shape[0], size=n_clusters, replace=False)
    centroids = dataset[indices]
    return centroids

def initialize_centroids_kpp(k, dataset):
    data = dataset.values
    centroids = []

    # primo centroide random
    centroids.append(data[np.random.randint(0, data.shape[0])])

    for _ in range(1, k):
        # distanza quadratica da ogni punto al centroide più vicino
        distances = np.array([
            min(np.linalg.norm(x - c)**2 for c in centroids)
            for x in data
        ])

        # probabilità proporzionale alla distanza
        probs = distances / distances.sum()

        # scegli il prossimo centroide
        idx = np.random.choice(data.shape[0], p=probs)
        centroids.append(data[idx])

    return np.array(centroids)

def update_centroids(cluster, n_clusters, dim_clusters):
    pass


def Lloyds(centroids, dataset, m, n_clusters, n_features, tol=1e-6):
    old_centroids = centroids.copy()
    new_centroids = np.zeros_like(centroids)

    print(old_centroids)
    print(new_centroids)
