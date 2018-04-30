import math
import numpy as np
import random
import sys
from utils import _point_distance, _cluster_distance

def KMeansClustering(data, k, seed_init=12345, large_output = False):
    """Clustering using K-Means algorithm
    
    Arguments:
        data {np.ndarray} -- (N * D) matrix where each entry is a D-vector
        k {int} -- Number of desired clusters
    
    Keyword Arguments:
        seed_init {int} -- Seed for random initialization of centroids (default: {12345})
        large_output {bool} -- If True, will return cluster and centroids objects (default: {False})
    
    Returns:
        np.ndarray -- (N * (D+1)) matrix where each entry has an additional column indicating the assigned cluster
    """

    random.seed(seed_init)
    D = len(data)
    centroids = []
    clusters = [[] for i in range(k)]
    point_cluster_mask = [0] * D
    n_changes = D

    # Select intial centroids
    centroid_numbers = set()
    while len(centroid_numbers) < k:
        r = random.randint(1,D-1)
        centroid_numbers.add(r)
    for i in centroid_numbers:
        centroids.append(data[i])

    while n_changes:                 
        # Reset clusters
        clusters = [[] for i in range(k)]

        # Calculate distances from centroids and assign to clusters
        n_changes = D
        distances = np.zeros((D, k))
        for d in range(D):
            for c in range(k):
                distances[d][c] = _point_distance(data[d], centroids[c])
            old_assignment = point_cluster_mask[d]
            assignment = np.argmin(distances[d])
            clusters[assignment].append(data[d])
            point_cluster_mask[d] = assignment
            if old_assignment == assignment:
                n_changes -= 1
        
        # Calculate new centroids
        for c in range(k):
            centroids[c] = np.sum(clusters[c], axis=0) / len(clusters[c])

    # Annotate data with cluster
    result = np.empty((D,3))
    for d in range(D):
        result[d] = np.append(data[d], point_cluster_mask[d])

    if large_output:
        return result, clusters, centroids, point_cluster_mask
    else:
        return result

def AggHierClustering(data, k, similarity_measure):
    """Agglomerative Hierarchical Clustering Algorithm
    
    Arguments:
        data {np.ndarray} -- (N * D) matrix where each entry is a D-vector
        k {int} -- number of desired clusters
        similarity_measure {str} -- The metric by which to compare similarity of clusters. See below for more detail.
    
    Returns:
        np.ndarray -- (N * (D+1)) matrix where each entry has an additional column indicating the assigned cluster)

    Similarity Measures:
        "min" - The distance between clusters is the distance between the two closest points in each cluster
        "max" - The distance between clusters is the distance between the two furthest points in each cluster
        "average" - The distance between clusters is the average of all distances (i,j) where i is in C1 and j is in C2
        "centroids" - The distance between clusters is the distance between the centroid of each cluster
    """

    D = len(data)
    clusters = [[d] for d in data]

    while len(clusters) > k:
        
        # Calculate distances between all clusters
        C = len(clusters)
        distances = np.ndarray((C, C))
        for i in range(C):
            for j in range(C):
                dist = _cluster_distance(clusters[i], clusters[j], similarity_measure)
                distances[i][j] = dist if (not i == j) else np.Infinity
        min_distance_locus = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

        # Merge two smallest clusters
        updated_clusters = []
        new_cluster = []
        for i in range(C):
            if (i == min_distance_locus[0] or i == min_distance_locus[1]):
                for d in clusters[i]:
                    new_cluster.append(d)
            else:
                updated_clusters.append(clusters[i])

        updated_clusters.append(new_cluster)
        clusters = updated_clusters
        
    
    # Annotate data with cluster
    result = np.empty((D,3))
    d = 0
    assert(len(clusters) == k)
    for i in range(k):
        for j in range(len(clusters[i])):
            annotated_point = np.append(clusters[i][j], i)
            result[d] = annotated_point
            d += 1              

    return result