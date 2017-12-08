
import numpy as np
import random
import math

def complearn_clustering(data, num_outputs, iterations, learning_rate = 0.1):

    weights = [np.random.uniform(0.0, 1, len(data[0])) for _ in range(num_outputs)]

    # train the network. If no change is made for stagnant iterations, training is terminated. Otherwise completes iterations
    stagnant = 5
    for i in range(iterations):
        # randomly select a point to test
        input = random.sample(data, 1)[0]
        # calculate the winner
        winner = compete(weights, input)
        # create the new weights for the winner
        new_weights = update_weights(weights, input, winner, learning_rate)
        # if no change was made, increment stagnant and check if training should terminate
        print('Iteration = {}, Winner = {}'.format(i, winner))
        print("Weights = {}".format(weights))
        if np.all(new_weights == weights[winner]):
            stagnant -= 1
            if stagnant <= 0: break
        # else update the weights, reset stagnant counter
        else:
            weights[winner] = new_weights
            stagnant = 5

    clusters = [[] for _ in range(num_outputs)]

    # now create the cluster on the trained network. Each point is added to the cluster specified by the node that wins
    for point in data:
        clusters[compete(weights, point)].append(point)

    new_clusters = []
    for cluster in clusters:
        if cluster: new_clusters.append(cluster)

    print('Number of clusters created = {}'.format(len(new_clusters)))
    print('CLUSTERS = {}'.format(new_clusters))

    return new_clusters

def compete(weights, input):
    ''' Determine which output nodes weights have the best match the input based on Euclidean distance.
        Return the id of the winning node. This method assumes the data is not normalized'''

    winner_id = -1
    winner_value = 99999
    # for each output node
    for i in range(len(weights)):
        # calculate its similarity to the input vector
        current_fitness = math.sqrt(sum([(a - b) ** 2 for a, b in zip(input, weights[i])]))
        # if the current node is more similar, update the current winner values
        if winner_value > current_fitness:
            winner_id = i
            winner_value = current_fitness

    return winner_id

def update_weights(weights, input, winner, learning_rate):
    ''' Create the updated weights for the winner using the old weights and the input. Assumes the
        data in not normalized. new_weight = weight + (feature - weight) for each weight. '''

    return [weight + learning_rate * (feature - weight) for weight, feature in zip(weights[winner], input)]


