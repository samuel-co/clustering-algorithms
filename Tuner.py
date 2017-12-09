

import Driver
import numpy as np

def tune_db(data, name, stepsize):

    mini, maxi = Driver.CompLearn.get_min_max_values(data)
    ranges = [maxi[i] - mini[i] for i in range(len(mini))]

    optimum_params = [0, 0]
    optimum_results = [1, 0, 99999]

    for radius in np.arange(min(ranges), max(ranges), (max(ranges) - min(ranges)) / stepsize):
        for  num_points in range(int(len(data)/stepsize), len(data), int(len(data)/stepsize) ):
            print('Num_points = {}, Radius = {}'.format(num_points, radius))
            clusters = Driver.DBScan.db_clustering(data, num_points, radius)
            results = Driver.test_clustering(clusters, 0)
            intra_improvement = optimum_results[0] - results[0]
            inter_improvement = results[1] - optimum_results[1]
            sse_improvement = 1 - results[2] / optimum_results[2]

            if intra_improvement + inter_improvement + sse_improvement > 0:
                print('updating results')
                optimum_results = results



if __name__ == "__main__":
    data, name = Driver.import_data('datasets/iris.txt')
    tune_db(data, name, 10)
