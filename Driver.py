
import os
import math
import KMeans
import DBScan
import CompLearn
import PSO
import ACO
import time
import copy
import random as rand
import matplotlib.pyplot as plt

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

def test_clustering(clusters, time, normal = False, output=True):
    ''' Print the evaluation metrics for a series of metrics. If normal = False, input clusters
        will be normalize before being evaluated. '''

    if not normal:
        extremes = [CompLearn.get_min_max_values(cluster) for cluster in clusters]
        mini, maxi = CompLearn.get_min_max_values([pair[0] for pair in extremes] + [pair[1] for pair in extremes])
        clusters = [CompLearn.normalize_data(cluster, mini, maxi) for cluster in clusters]

    # calculate the evaluation metrics
    eval = [intra_distance(clusters), inter_distance(clusters), SSE(clusters)]

    # If the method is expected to output the info, do so
    if output:
        print("- Number of clusters = {}".format(len(clusters)))
        print("- Intra-distance = {}".format(eval[0]))
        print("- Inter-distance = {}".format(eval[1]))
        print("- SSE = {}".format(eval[2]))
        print("- Time elapsed = {}\n".format(time))

    return eval

def show_2d_clusters(data, clusters, normal=False):

    if not normal:
        mini, maxi = CompLearn.get_min_max_values(data)
        clusters = [CompLearn.normalize_data(cluster, mini, maxi) for cluster in clusters]

    colors = {0: 'k', 1: 'b', 2: 'g', 3: 'r', 4: 'c', 5: 'y', 6: 'm'}

    plt.scatter([point[0] for point in data], [point[1] for point in data], c='c', alpha=.2)

    for i in range(len(data[0])-1):
        for j in range(i+1, len(data[0])):
            color = 0
            plt.title("Plotting features {}, {}".format(i, j))
            plt.xlabel("Feature {}".format(i))
            plt.ylabel("Feature {}".format(j))
            plt.grid(True)
            for cluster in clusters:
                plt.scatter([point[i] for point in cluster], [point[j] for point in cluster], c=colors[color % 7])
                color += 1
            plt.show()

def main():

    mockData = []
    for i in range(100):
        point = []
        for j in range(2):
            val = rand.uniform(0, .4)
            point.append(val)
        mockData.append(point)
        point = []
        for j in range(2):
            val = rand.uniform(.6, 1)
            point.append(val)
        mockData.append(point)

        point = []
        point.append(rand.uniform(0, .4))
        point.append(rand.uniform(0.6, 1))
        #mockData.append(point)

        point = []
        point.append(rand.uniform(.6, 1))
        point.append(rand.uniform(0, 0.4))
        #mockData.append(point)




    data, name = import_data('datasets/car.txt')  # 4 clusters
    data, name = import_data('datasets/cmc.txt')  # 3 clusters
    data, name = import_data('datasets/yeast.txt')  # 5/6 to 10 clusters
    data, name = import_data('datasets/wholesale_customers data.txt') # ? clusters
    #data, name = (mockData, 'mock')
    data, name = import_data('datasets/iris.txt')  # 3 clusters
    #data, name = import_data('datasets/seeds.txt')  # 3 clusters

    if False:
        print("Using K-Means to cluster dataset {}:".format(name))
        start = time.time()
        clusters = KMeans.kmeans_clustering(copy.deepcopy(data), 4, 10000)
        end = time.time()
        test_clustering(clusters, end-start, normal=False)

        #show_2d_clusters(data, clusters, normal=False)

    if False:
        print("Using DBScan to cluster dataset {}:".format(name))
        start = time.time()
        clusters = DBScan.db_clustering(copy.deepcopy(data), 30, 1.6)
        end = time.time()
        if clusters:
            test_clustering(clusters, end - start, normal=False)

        #show_2d_clusters(data, clusters, normal=False)

    if False:
        print("Using CompLearn to cluster dataset {}:".format(name))
        start = time.time()
        clusters = CompLearn.complearn_clustering(copy.deepcopy(data), 6, 100000)
        end = time.time()
        test_clustering(clusters, end - start, normal=True)

        #show_2d_clusters(data, clusters, normal=True)

    if False:
        print("Using PSO to cluster dataset {}:".format(name))
        start = time.time()
        clusters = PSO.pso_clustering(copy.deepcopy(data), 3, 1000)
        end = time.time()
        test_clustering(clusters, end - start, normal=False)

        #show_2d_clusters(data, clusters, normal=False)

    if True:
        print("Using ACO to cluster dataset {}:".format(name))
        start = time.time()
        clusters = ACO.aco_clustering(copy.deepcopy(data), 40, 100000)
        end = time.time()
        test_clustering(clusters, end - start, normal=True)


        #show_2d_clusters(data, clusters, normal=False)



if __name__ == '__main__':
    main()