import numpy as np
import joblib
import tasklogger
import sklearn.cluster
import sklearn.neighbors
import scipy.spatial.distance


def get_compression_features(N, features, n_pca):
    """Short summary.

    Parameters
    ----------
    N : type
        Description of parameter `N`.
    features : type
        Description of parameter `features`.
    n_pca : type
        Description of parameter `n_pca`.

    Returns
    -------
    type
        Description of returned object.

    """
    if n_pca == None:
        n_pca = min(N, features)
    if n_pca > 100:
        n_pca = 100

        n_pca = 100

    return n_pca

def merge_clusters(diff_pot_unmerged, clusters):
    """Short summary.

    Parameters
    ----------
    diff_pot_unmerged : type
        Description of parameter `diff_pot_unmerged`.
    clusters : type
        Description of parameter `clusters`.

    Returns
    -------
    type
        Description of returned object.

    """
    clusters_uni = np.unique(clusters)
    num_clusters = len(clusters_uni)
    diff_pot_merged = np.zeros(num_clusters * diff_pot_unmerged.shape[1]).reshape(
        num_clusters, diff_pot_unmerged.shape[1]
    )

    for c in range(num_clusters):
        loc = np.where(clusters_uni[c] == clusters)[0]
        diff_pot_merged[c, :] = np.nanmean(diff_pot_unmerged[loc], axis=0)

    return diff_pot_merged


def get_distance_from_centroids(centroids, data, clusters):
    distance = np.zeros(centroids.shape[0])

    for c in range(centroids.shape[0]):
        cluster_points = data[clusters == c]
        dist = []

        for i in range(cluster_points.shape[0]):
            dist.append(
                scipy.spatial.distance.sqeuclidean(
                    centroids[c, :], cluster_points[i, :]
                )
            )
        distance[c] = np.max(dist)
    return distance


def map_update_data(centroids, data, new_data, partition_clusters, nn=5, n_jobs=10):
    """Short summary.

    Parameters
    ----------
    centroids : type
        Description of parameter `centroids`.
    data : type
        Description of parameter `data`.
    new_data : type
        Description of parameter `new_data`.
    partition_clusters : type
        Description of parameter `partition_clusters`.
    nn : type
        Description of parameter `nn`.
    n_jobs : type
        Description of parameter `n_jobs`.

    Returns
    -------
    type
        Description of returned object.

    """
    with tasklogger.log_task("map to computed partitions"):
        # getting max distance to each partition centroid
        distance_merged = get_distance_from_centroids(
            centroids, data, partition_clusters
        )

        # Mapping NN in new data to centroids
        NN_op = sklearn.neighbors.NearestNeighbors(n_neighbors=nn, n_jobs=n_jobs)
        NN_op.fit(centroids)
        neighbor_dists, neighbor_idx = NN_op.kneighbors(new_data)

        # Identifying which new data points fall below threshold
        parition_assignment_bool = neighbor_dists < distance_merged[neighbor_idx]

        subset_partition_assignment = np.zeros(new_data.shape[0])
        subset_partition_assignment[subset_partition_assignment == 0] = -1

        # Finding neatest mapped partition centroid
        for r in range(len(subset_partition_assignment)):
            c = 0
            while c < nn:
                if parition_assignment_bool[r, c] == True:
                    subset_partition_assignment[r] = neighbor_idx[r, c]
                    c = nn + 1
                    break
                else:
                    c += 1

    return subset_partition_assignment
