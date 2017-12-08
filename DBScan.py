'''
Sam Congdon, Kendall Dilorenzo, Michel Hewitt
CSCI 447: MachineLearning
Project 4: DB-Scan
December 11, 2017

This python module is used to create a series of clusters using the DB-Scan algorithm. db_clustering() is the
primary function, find_border and expand_cluster are helper functions. Requires data points to be clustered
and the minimum number of local points for a core as parameters. Radius is an optional parameter, it can be
calculated if not supplied. Returns a list of clusters, each cluster represented as a list of the points contained.

NOTE: This line of code is used to calculate the Euclidean distance between points x and y of any matching dimensions
math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
'''

import math

def db_clustering(data, num_points, radius=None):
    ''' Create a list of clusters through the DBScan algorithm. Parameters are the data points to be clustered,
        the minimum number of neighboring points required to create a core point, and the radius to search for
        these neighbors. The algorithm iterates over each data point, checking if it can be a core point. If so,
        the point is passed to the expand cluster function. This function will grow the cluster by analyzing each
        neighboring point to determine if it's of adequate density to be in the cluster. If so, this point is added,
        then all of its neighbors are evaluated for the same criteria. This is performed until no more neighbors
        are left to evaluate. Returns a list of clusters, each cluster represented as a list of the points contained. '''

    clusters = []

    # if no radius is defined, define the radius as .2 of the maximum euclidean distance present in the data
    if radius is None:
        radius = 0
        for x in data:
            for y in data:
                radius = max(radius, math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))
        radius = 0.2 * radius

    # create an entry to mark the points with, initially as unvisited
    for point in data: point.append(None)

    # check each data point
    for point in data:
        # if the point has already been visited, do nothing to it
        if point[-1] is not None: pass
        # else check if this point can be a core point
        else:
            # get a list of neighbors
            neighbors = find_border(data, point, radius)
            # if this is a core point
            if len(neighbors) >= num_points:
                # label it as core and build a cluster
                point[-1] = 'C'
                clusters.append(expand_cluster(data, point, neighbors, radius, num_points))
            # otherwise label the point as noise
            else: point[-1] = 'N'

    #print('DATA = {}'.format(data))

    # delete the point labels added to perform DBScan
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            del clusters[i][j][-1]

    #print('Number of clusters created = {}'.format(len(clusters)))
    #print('CLUSTERS = {}'.format(clusters))

    return clusters

def find_border(data, point, radius):
    ''' Find and return all points within the radius of our passed in point '''

    neighbors = []
    # for each point in the data, check its distance from the passed in point.
    for candidate in data:
        # if the point is within range, add it to the list of neighbors
        if candidate != point and math.sqrt(sum([(a - b) ** 2 for a, b in zip(candidate[:-1], point[:-1])])) < radius:
            neighbors.append(candidate)
    return neighbors

def expand_cluster(data, core_point, neighbors, radius, num_points):
    ''' Creates a cluster based on a core point. The initial list of neighbors is passed in to avoid
        duplicate operations being performed '''
    cluster = [core_point]

    # as long as point remain to be checked
    while neighbors:
        point = neighbors.pop(0)
        # if this point has not yet been visited
        if point[-1] is None:
            # check if the point has an adequate number of neighbors to act as a core point.
            candidates = find_border(data, point, radius)
            if len(candidates) >= num_points:
                # if enough candidates are neighboring this neighbor, add them to our list of neighbors
                neighbors.append(candidates)
            # label the current point and add it to the cluster
            point[-1] = 'B'
            cluster.append(point)

    return cluster

