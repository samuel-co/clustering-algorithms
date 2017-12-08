
import os
import math
import KMeans
import DBScan
import CompLearn
import time

def import_data(file_name):
    ''' Imports data points from supplied file, formatting them into data point lists to be clustered. '''

    fin = open(file_name, 'r')
    input_line = fin.readline()
    data = []

    while input_line:
        input_line = input_line.strip().split(',')
        if input_line == ['']: break
        for i in range(len(input_line)): input_line[i] = float(input_line[i])
        data.append((input_line))
        input_line = fin.readline()
    fin.close()

    return data, os.path.splitext(file_name)[0][9:] # last return is file name, assumes file is in datasets/ directory

def intra_distance(clusters):
    ''' Calculate the average intra-cluster distance for every cluster contained in clusters. Calculated by
        averaging the squared Euclidean distance between every point within each cluster. Want to minimize.'''

    total_sum = 0
    for cluster in clusters:
        cluster_sum = 0
        for x in cluster:
            for y in cluster:
                if x != y:
                    cluster_sum += math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])) ** 2
        total_sum += cluster_sum / len(cluster) ** 2
    return total_sum / len(clusters)

def inter_distance(clusters):
    ''' Calculate the average inter-cluster distance for each pair of clusters in clusters. Calculated as
        the average squared Euclidean distance between each point of each pair of clusters. Want to maximize. '''

    total_sum = 0
    for c1 in clusters:
        for c2 in clusters:
            inner_sum = 0
            if c1 != c2:
                for x in c1:
                    for y in c2:
                        inner_sum += math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])) ** 2
                total_sum += inner_sum / (len(c1) * len(c2))
    return total_sum / len(clusters) ** 2


def SSE(clusters):
    ''' Calculate the Sum Square Error (SSE) of the clusters. Calculated as the sum of each points
        squared Euclidean distance to the mean of the cluster its contained in. Want to minimize. '''

    total_sum = 0
    for cluster in clusters:
        inner_sum = 0

        # calulate the mean value of each feature over all the points in the cluster, used to create the mean_point
        feature_sums = []
        for i in range(len(cluster[0])):
            feature_sums += [sum([point[i] for point in cluster])]
        mean_point = [feature_sum/len(cluster) for feature_sum in feature_sums]

        # sum the SSE of each point in the cluster
        for point in cluster:
            inner_sum += math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, mean_point)])) ** 2
        total_sum += inner_sum

    return total_sum


def main():
    data, name = import_data('datasets/iris.txt')

    print("Using K-Means to cluster dataset {}:".format(name))
    start = time.time()
    clusters = KMeans.kmeans_clustering(data, 3, 10000)
    end = time.time()
    mini, maxi = CompLearn.get_min_max_values(data)
    normal_clusters = [CompLearn.normalize_data(cluster, mini, maxi) for cluster in clusters]
    print("- Number of clusters = {}".format(len(clusters)))
    print("- Intra-distance = {}".format(intra_distance(normal_clusters)))
    print("- Inter-distance = {}".format(inter_distance(normal_clusters)))
    print("- SSE = {}".format(SSE(normal_clusters)))
    print("- Time elapsed = {}\n".format(end - start))

    print("Using DBScan to cluster dataset {}:".format(name))
    start = time.time()
    clusters = DBScan.db_clustering(data, 30, 2.6)
    end = time.time()
    mini, maxi = CompLearn.get_min_max_values(data)
    normal_clusters = [CompLearn.normalize_data(cluster, mini, maxi) for cluster in clusters]
    print("- Number of clusters = {}".format(len(clusters)))
    print("- Intra-distance = {}".format(intra_distance(normal_clusters)))
    print("- Inter-distance = {}".format(inter_distance(normal_clusters)))
    print("- SSE = {}".format(SSE(normal_clusters)))
    print("- Time elapsed = {}\n".format(end - start))

    print("Using CompLearn to cluster dataset {}:".format(name))
    start = time.time()
    normal_clusters = CompLearn.complearn_clustering(data, 6, 100000)
    end = time.time()
    print("- Number of clusters = {}".format(len(normal_clusters)))
    print("- Intra-distance = {}".format(intra_distance(normal_clusters)))
    print("- Inter-distance = {}".format(inter_distance(normal_clusters)))
    print("- SSE = {}".format(SSE(normal_clusters)))
    print("- Time elapsed = {}\n".format(end-start))


if __name__ == '__main__':
    main()