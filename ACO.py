'''
Sam Congdon, Kendall Dilorenzo, Micheal Hewitt
CSCI 447: MachineLearning
Project 4: ACO
December 11, 2017

This python module is used to create a series of clusters using a ACO algorithm. The Ant class handles
the creation and actions of individual ants within the algorithm. It tracks ants info, such as position,
as well as performs functionality such as movement. The primary function is aco_clustering(), which handles
the data and Ant class to create the clusters. Parameters are the data to be clustered, the number of ants
to implement, and the maximum number of training iterations to complete. Returns a list of clusters, each
cluster represented as a list of the points contained.
'''

import random
import math
import copy
import CompLearn
import matplotlib.pyplot as plt

class Ant:
    ''' The Ant class handles the creation and actions of individual ants within the algorithm. It tracks
        ants info, such as position, as well as performs functionality such as movement. '''

    def __init__(self, dimensions):
        ''' Initializes info for each ant. Parameter is the dimensions of the grid. '''

        # place the ant at a random position on the grid
        self.position = [random.randint(0, dimensions) for _ in range(2)]
        self.dimensions = dimensions
        self.item = None
        self.step_size = 2
        self.search_radius = 3

    def decide_action(self, grid, terminate=False):
        ''' Manages an ants action each turn, i.e. chooses whether it will try to pick up an item, drop an
            item, or just move. If an ant fails at picking up or dropping, it then moves. Setting terminate
            to true prevents ants from picking up items, allowing the algorithm to end. '''

        #print("\nDeciding an action...")
        # try to pick an item up
        if self.item is None and grid[tuple(self.position)] is not None and not terminate:
            #print("Attempting pickup")
            self.pickup(grid)

            # if we failed at picking up, move
            if self.item is None:
                self.move()
            # else remove the item from the grid
            else:
                return None

        # try to drop an item
        elif self.item is not None and grid[tuple(self.position)] is None:
            #print("Attempting drop")
            result = self.drop(grid)

            # if we failed at dropping, move the ant
            if self.item is not None:
                self.move()
            # else place the item back into the grid
            else:
                return result

        # else just move, update the grid with its current value
        else: self.move()
        return grid[tuple(self.position)]


    def pickup(self, grid):
        ''' Try and pick an item up off of the grid based on surroundings fitness. '''

        pickup_probability = self.evaluate_fitness(grid, grid[tuple(self.position)])
        # if no other points are within our range, give ourselves a high chance of picking up the object
        if pickup_probability == 0: pickup_probability = 0.1

        # higher the probability, the less the item fits there
        if  pickup_probability > random.random():
            #print("Picking up item")
            # if we manage to pick it up, update the ants item and return the new grid value
            self.item = copy.deepcopy(grid[tuple(self.position)])
            return None

        # we didn't pick it up, no change in the grid
        return grid[tuple(self.position)]


    def drop(self, grid):
        ''' Try and drop an item onto the grid based on surroundings fitness. '''

        drop_probability = self.evaluate_fitness(grid, self.item)
        # if theres no other points within sight, give ourselves a very low chance of dropping the item
        if drop_probability == 0: drop_probability = 0.99

        # higher the probability, the less the item fits there
        if drop_probability < random.random():
            # if we succeed at dropping, update the ant and return the item placed on the grid
            #print("Dropping item")
            item = copy.deepcopy(self.item)
            self.item = None
            return item

        # otherwise no update to the grid
        return None


    def move(self):
        ''' Move an ant in a random direction a random amount within our set step_size. '''

        #print("Moving ant from {} to ".format(self.position), end='')
        # loop until a valid move is found
        while True:
            x = random.choice(range(self.position[0] - self.step_size, self.position[0] + self.step_size + 1))
            y = random.choice(range(self.position[1] - self.step_size, self.position[1] + self.step_size + 1))
            if x in range(self.dimensions) and y in range(self.dimensions) and (x != self.position[0] or y != self.position[1]):
                self.position = [x, y]
                break
        #print("{}".format(self.position))

    def evaluate_fitness(self, grid, item):
        ''' Evaluate the fitness of a passed in item based on the ants current location on the grid. '''
        #print("Checking fitness of point {} with: ".format(item))

        # create an array of Euclidean distances to all neighboring points
        distances = []
        # loop through all positions within our search radius
        for x in range(self.position[0] - self.search_radius, self.position[0] + self.search_radius + 1):
            for y in range(self.position[1] - self.search_radius, self.position[1] + self.search_radius + 1):
                # if the position is valid and there's a point on it, append it's Euclidean distance
                if x in range(self.dimensions) and y in range(self.dimensions) and [x, y] != self.position and grid[(x, y)] is not None:
                    #print(grid[(x, y)])
                    distances.append(math.sqrt(sum([(a - b) ** 2 for a, b in zip(grid[(x, y)], item)])))

        #fitness = None
        # our fitness is the average distance to all located points
        #if len(distances) > 0: fitness = sum(distances) / len(distances)

        # fitness is the sum euclidean's over the number of squares we searched, accounting for density
        fitness = 0
        if len(distances) > 0: fitness = sum(distances) / ((2 * self.search_radius) ** 2 - 1)
        #print("Final fitness = {}".format(fitness))
        return fitness

    def place(self, grid):
        ''' Force an ant to place its item once all training iterations have been completed. The ant gets to search
            through the entire space and place the item in the optimum point. Time consuming, its faster to use
            terminate within decide actions, however the placement wont be optimal.

             NOTE: place() was not used during the experimental tests. '''

        # if the ant has no item, quit
        if self.item is None:
            return grid

        best_fitness = 99999
        best_position = [0, 0]
        # loop over every position in the grid
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                # if this is a valid placement point
                if grid[x, y] is None:
                    # evaluate the fitness here
                    current_fitness = self.evaluate_fitness(grid, self.item)
                    # if this fitness is better, save it and the position
                    if current_fitness is not None and current_fitness < best_fitness:
                        best_fitness = current_fitness
                        best_position = [x, y]

        #print("Placing value at {}".format(best_position))
        # place the item at the best possible point and return the grid
        grid[tuple(best_position)] = copy.deepcopy(self.item)
        return grid


def aco_clustering(data, num_ants, iterations=1000, show_plots=False):
    ''' Create a series of clusters through the ACO algorithm. Parameters are the data to be clustered, the
        number of ants to use, and the maximum number of training iterations to complete. Creates a 2D grid
        to randomly place the data points over, then creates the specified number of ants and randomly places
        them on the grid as well. The ants travel around moving data points they found to be near similar data
        points, similarity determined by Euclidean distance. Once the ants have completed their placement,
        clusters are created through applying an adjacency clustering method to the points as they were placed
        over the 2D grid by the ants. '''

    # normalize the data for easy determination of pickup and drop probabilities
    mini, maxi = CompLearn.get_min_max_values(data)
    data = CompLearn.normalize_data(data, mini, maxi)

    # calculate the dimensions of the grid from the size of the data
    #dimensions = int(math.sqrt(10 * len(data))) - 1
    dimensions = len(data) - 1

    # create the initially empty grid
    grid = {}
    for i in range(dimensions + 1):
        for j in range(dimensions + 1):
            grid.update({(i, j) : None})

    # place our data randomly over the grid
    for point in data:
        # loop until a valid placement is found
        while True:
            placement = (random.randint(0, dimensions), random.randint(0, dimensions))
            if grid[placement] == None:
                grid[placement] = point
                break

    # create our ants, beginning at random positions
    ants = []
    for ant in range(num_ants):
        ants.append(Ant(dimensions))

    # train for the set number of iterations
    for i in range(iterations+1):
        # each ant performs a task every training iteration
        for ant in ants:
            grid[tuple(ant.position)] = ant.decide_action(grid)

        # prints out the current clustering state for the mock data, only works on the mock data
        if show_plots and i%int(iterations / 10) == 0:
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            for key, value in grid.items():
                if value is not None:
                    if value[0] > 0.5:
                        x1.append(key[0])
                        y1.append(key[1])
                    else:
                        x2.append(key[0])
                        y2.append(key[1])
            plt.title("Grid at iteration {}".format(i))
            plt.scatter(x1, y1, c='b', s=[1 for _ in range(len(x1))], alpha=0.5)
            plt.scatter(x2, y2, c='r', s=[1 for _ in range(len(x2))], alpha=0.5)
            plt.show()

    # force all the ants to drop their items, delete them once their not carrying anything
    radius = ants[0].search_radius
    while ants:
        for ant in ants:
                #grid = ant.place(grid)
                grid[tuple(ant.position)] = ant.decide_action(grid, terminate=True)
                if ant.item is None: ants.remove(ant)

    clusters = find_clusters(grid, dimensions, radius, join_clusters=False)
    return clusters

def find_clusters(grid, dimensions, search_radius, join_clusters=False):
    ''' This method uses an adjacency clustering technique to build clusters based on data point
        locations over the grid manipulated by the ants. join_clusters can be set to True in order
        to join the clusters made by the ants. NOT RECOMMENDED'''

    # create a 2D list specifying where points lie in the grid, points are specified with a 1, empty places with a 0
    graph = [[0 for _ in range(dimensions + 1)] for _ in range(dimensions + 1)]
    for key, value in grid.items():
        key = list(key)
        if value is not None:
            graph[key[0]][key[1]] = 1

    # create an array specifying which points have been visited in our graph, using False for not visited
    visited = [[False for _ in range(dimensions + 1)] for _ in range(dimensions + 1)]

    # check every point in the graph, if it hasn't already been checked and has a value, create an island from it
    islands = []
    for key, value in grid.items():
        key = list(key)
        if not visited[key[0]][key[1]]:
            visited[key[0]][key[1]] = True
            if graph[key[0]][key[1]] == 1:
                islands.append(DFS(graph, visited, key, dimensions, search_radius) + [key])

    # change the island values from coordinates to the actual points
    for i in range(len(islands)):
        islands[i] = [grid[tuple(position)] for position in islands[i]]

    # if we don't want to join the ants cluster and just use their results, return now
    if not join_clusters:
        return islands

    #----------------------The following code was not used during the experiments--------------------------------

    # now organize the islands by their respective sizes
    island_sizes = [len(island) for island in islands]
    biggest_cluster = max(island_sizes)
    island_sizes = [[size, island] for size, island in zip(island_sizes, islands)]
    island_sizes.sort()

    # the largest island will always seed a cluster
    clusters = [island_sizes[-1][1]]
    island_sizes = island_sizes[:-1]
    minnis = []

    # if islands are too small we'll add them to other clusters, otherwise they can be a cluster
    for island in reversed(island_sizes):
        if island[0] < dimensions / len(islands):
            minnis.append(island[1])
        else:
            clusters.append(island[1])

    # if there were no tiny clusters, return now
    if len(minnis) < 1:
        return clusters

    # calculate the mean point of each cluster
    cluster_means = []
    for cluster in clusters:
        feature_sums = []
        for i in range(len(cluster[0])):
            feature_sums += [sum([point[i] for point in cluster])]
        cluster_means.append([feature_sum / len(cluster) for feature_sum in feature_sums])

    # calculate the mean point of each mini
    mini_means = []
    for mini in minnis:
        feature_sums = []
        for i in range(len(mini[0])):
            feature_sums += [sum([point[i] for point in mini])]
        mini_means.append([feature_sum / len(mini) for feature_sum in feature_sums])

    # for each island that was too small to be a cluster
    for i in range(len(minnis)):
        smallest = 99999
        id = -1
        # find the cluster that has the closest mean
        for j in range(len(cluster_means)):
            dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(mini_means[i], cluster_means[j])]))
            if  dist < smallest:
                smallest = dist
                id = j
        # add this cluster to the closest found cluster
        clusters[id] += minnis[i]

    return clusters


def DFS(graph, visited, position, dimensions, search_radius):
    ''' Find all of the data points lieing within our search radius on the grid. Utilizes Depth First Search
        to recursively find all the neighboring neighbors in order to create the island. Returns the completed
        island. '''

    # create an array of neighboring points to the passed in point
    neighbors = []
    # loop through all positions within our search radius
    for x in range(position[0] - search_radius, position[0] + search_radius + 1):
        for y in range(position[1] - search_radius, position[1] + search_radius + 1):
            # if the position is valid and there's and it hasnt been visited, mark it as True
            if x in range(dimensions+1) and y in range(dimensions+1) and [x, y] != position and visited[x][y] is not True:
                visited[x][y] = True
                # if the position has a point in it, add it and its neighbors to our neighbors
                if graph[x][y] == 1:
                    neighbors.append([x, y])
                    neighbors += DFS(graph, visited, [x, y], dimensions, search_radius)

    return neighbors


