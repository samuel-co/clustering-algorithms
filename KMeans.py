'''
Sam Congdon, Kendall Dilorenzo, Micheal Hewitt
CSCI 447: MachineLearning
Project 4: KMeans
December 11, 2017

This python module is used to create a series of clusters using a k-means clustering. kmeans_clustering()
is the primary function, find_clusters and move_centroids are helper functions. Parameters are the data to
be clustered, the number of clusters to be created, and the maximum number of iterations to perform before
termination. Returns a list of clusters, each cluster represented as a list of the points contained.
'''


import numpy as np
import math
import CompLearn

def kmeans_clustering(data, k, iterations):
    ''' Create a list of clusters using k-means clustering. Parameters are the data to be clustered,
        the number of clusters to be created, and the maximum number of iterations to perform before
        termination. The training will terminate early if a stable state is achieved. Clusters are
        built from the current centroids, then the location of the centroids is updated to be the
        mean of the points in the centroid's cluster. Returns a list of clusters, each cluster
        represented as a list of the points contained.'''

    # randomly place our centroids area of our data, not necessarily on any point
    centroids = []
    mini, maxi = CompLearn.get_min_max_values(data)
    for i in range(k):
        centroids.append(np.array([np.random.uniform(mini[i], maxi[i], 1) for i in range(len(mini))]).flatten())

    # limit the maximum number of iterations that can be run
    for i in range(iterations):
        # create the current clusters
        clusters = find_clusters(data, k, centroids)
        # update the locations of the centroids
        new_centroids = move_centroids(clusters, centroids.copy(), mini, maxi)
        # if the centroid locations have not changed, terminate the algorithm
        if np.all(np.array(new_centroids) == centroids):
            print("KMeans terminated after {} iterations as stagnation was achieved".format(i))
            break
        centroids = new_centroids

    return find_clusters(data, k, centroids)


def find_clusters(data, k, centroids):
    ''' Find the current clusters based on the min distance between each point and centroid. Returns
        the list of clusters. '''

    # create the current clusters
    clusters = [[] for _ in range(k)]
    # assign each point in the data
    for point in data:
        mini = 99999
        cluster = -1
        # check the distance to each cluster
        for i in range(len(centroids)):
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, centroids[i])]))
            # if the distance is smaller, update the cluster assignment
            if distance < mini:
                mini = distance
                cluster = i
        # assign the point to the closest found cluster
        clusters[cluster].append(point)

    return clusters

def move_centroids(clusters, centroids, mini, maxi):
    ''' Move the centroids to be the mean of all the points currently contained in their cluster. If
        a centroid has no points, randomly reinitialize it within the space. '''

    # move each centroid
    for i in range(len(centroids)):
        # if the centroid did not get any points in its cluster, randomly reinitialize it
        if not clusters[i]:
            centroids[i] = np.array([np.random.uniform(mini[i], maxi[i], 1) for i in range(len(mini))]).flatten()
        # else move the centroid to be the mean of all the points in its cluster
        else:
            feature_sums = []
            for j in range(len(clusters[i][0])):
                feature_sums += [sum([point[j] for point in clusters[i]])]
            centroids[i] = [feature_sum / len(clusters[i]) for feature_sum in feature_sums]

    return centroids
