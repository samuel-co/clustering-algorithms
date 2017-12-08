
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
        cluster_sum = []
        for i in range(len(cluster)-1):
            for j in range(i+1, len(cluster)):
                cluster_sum += [math.sqrt(sum([(a - b) ** 2 for a, b in zip(cluster[i], cluster[j])])) ** 2]
        if len(cluster_sum) > 0: total_sum += sum(cluster_sum) / len(cluster_sum)
    return total_sum / len(clusters)


def inter_distance(clusters):
    ''' Calculate the average inter-cluster distance for each pair of clusters in clusters. Calculated as
           the average squared Euclidean distance between each point of each pair of clusters. Want to maximize. '''
    if len(clusters) == 1:
        return 0

    total_sum = []
    for c1 in range(len(clusters)-1):
        for c2 in range(c1+1, len(clusters)):
            inner_sum = []
            for x in clusters[c1]:
                for y in clusters[c2]:
                    inner_sum += [math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])) ** 2]
            total_sum += [sum(inner_sum) / len(inner_sum)]
    return sum(total_sum) / len(total_sum)

def SSE(clusters):
    ''' Calculate the Sum Square Error (SSE) of the clusters. Calculated as the sum of each points
        squared Euclidean distance to the mean of the cluster its contained in. Want to minimize. '''

    total_sum = 0
    for cluster in clusters:
        # calulate the mean value of each feature over all the points in the cluster, used to create the mean_point
        feature_sums = []
        for i in range(len(cluster[0])):
            feature_sums += [sum([point[i] for point in cluster])]
        mean_point = [feature_sum/len(cluster) for feature_sum in feature_sums]

        # sum the SSE of each point in the cluster
        for point in cluster:
            total_sum += math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, mean_point)])) ** 2

    return total_sum

def test_clustering(clusters, time, normal = False):
    ''' Print the evaluation metrics for a series of metrics. If normal = False, input clusters
        will be normalize before being evaluated. '''

    if not normal:
        extremes = [CompLearn.get_min_max_values(cluster) for cluster in clusters]
        mini, temp = CompLearn.get_min_max_values([pair[0] for pair in extremes])
        temp, maxi = CompLearn.get_min_max_values([pair[1] for pair in extremes])
        clusters = [CompLearn.normalize_data(cluster, mini, maxi) for cluster in clusters]

    print("- Number of clusters = {}".format(len(clusters)))
    print("- Intra-distance = {}".format(intra_distance(clusters)))
    print("- Inter-distance = {}".format(inter_distance(clusters)))
    print("- SSE = {}".format(SSE(clusters)))
    print("- Time elapsed = {}\n".format(time))


def main():
    data, name = import_data('datasets/iris.txt')

    print("Using K-Means to cluster dataset {}:".format(name))
    start = time.time()
    clusters = KMeans.kmeans_clustering(data, 3, 10000)
    end = time.time()
    test_clustering(clusters, end-start)


    print("Using DBScan to cluster dataset {}:".format(name))
    start = time.time()
    clusters = DBScan.db_clustering(data, 30, 2.6)
    end = time.time()
    end = time.time()
    test_clustering(clusters, end - start)


    print("Using CompLearn to cluster dataset {}:".format(name))
    start = time.time()
    clusters = CompLearn.complearn_clustering(data, 6, 100000)
    end = time.time()
    end = time.time()
    test_clustering(clusters, end - start, True)


if __name__ == '__main__':
    main()