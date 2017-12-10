'''
Sam Congdon, Kendall Dilorenzo, Micheal Hewitt
CSCI 447: MachineLearning
Project 4: PSO
December 11, 2017

This python module is used to create a series of clusters using a PSO algorithm. The Particle class is
used to track each particles information, such as its velocity and position history. The primary function
is pso_clustering(), which handles the data and particle class to create the clusters. Parameters are the
data, number of clusters to create, and maximum number of iterations to perform. Will terminate if a stable
state is achieved. Returns a list of clusters, each cluster represented as a list of the points contained.
'''


import random
import copy
import math


class Particle:
    ''' The Particle class handle particle object, tracking each particles relevant information such as its
        best positions and fitness and the current position and velocity. Methods are for initialization,
        evaluating the particles fitness at its current position, updating the particle's velocity, and moving
        the particle. '''

    def __init__(self, dimensions, number_clusters):
        ''' Intializes the info for a particle. Parameters are the dimensions of the data, and the number of
            clusters that are to be created. '''

        # create a position entry for each centroid, then  create a velocity vector of the same size.
        self.position = [random.random() for _ in range(dimensions * number_clusters)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions * number_clusters)]
        self.best_position = copy.deepcopy(self.position)
        # arbitrary initial fitness values
        self.fitness = 0
        self.best_fitness = 99999
        self.dimensions = dimensions


    def evaluate_fitness(self, data):
        ''' Evaluate the fitness for this particle at its current position. Fitness is evaluated as the sum of
            the Euclidean distances from each point to its nearest centroid. '''

        self.fitness = 0
        # evaluate every point
        for point in data:
            fitnesses = []
            # calculate the difference from the point to each centroid we have in our position
            for i in range(int(len(self.position) / self.dimensions)):
                fitnesses.append(math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, self.position[i*self.dimensions:(i+1) * self.dimensions])])))
            # minimum distance to the fitness
            self.fitness += min(fitnesses)

        # if we've reached a new optimum for this particle, update the best memories
        if self.fitness < self.best_fitness:
            self.best_fitness = copy.deepcopy(self.fitness)
            self.best_position = copy.deepcopy(self.position)


    def update_velocity(self, gb_location):
        ''' Calculates the new velocity for each point based on the current best global location,
            the particles best recorded location, and the current velocity value. '''

        # for every velocity value
        for i in range(len(self.velocity)):

            # calculate and update with the new velocity value
            self.velocity[i] = self.velocity[i] + random.random() * (self.best_position[i] - self.position[i]) + random.random() * (gb_location[i] - self.position[i])

            # clamping method to keep the value below 100
            while math.fabs(self.velocity[i]) > 100:
                self.velocity[i] = self.velocity[i] / 2


    def move(self):
        ''' Moves the particle from its current position based on its current velocities. '''

        # update each position value using its relevant velocity
        for i in range(len(self.velocity)):
            self.position[i] += self.velocity[i]


def pso_clustering(data, number_clusters, iterations=1000):
    ''' Create a list of clusters through the PSO algorithm. Parameters are the data, number of clusters
        to create, and maximum number of iterations to perform. Will terminate if a stable state is achieved.
        Creates an equal number of particles as there are points in the data set. Continually iterates through
        the list of particles, updating their positions and tracking the best position found yet. Returns a
        list of clusters, each cluster represented as a list of the points contained.'''

    # initialize particles
    particles = []
    for _ in range(len(data)):
        particles.append(Particle(len(data[0]), number_clusters))

    # arbitrary initial fitness and position
    gb_fitness = 99999
    gb_position = particles[0].position

    # maximum number of stable iterations needed for early termination
    stagnant = 500
    for i in range(iterations):

        # update each particle within the swarm
        for particle in particles:
            particle.evaluate_fitness(data)

            # if we've found a better position, update our global bests
            if particle.fitness < gb_fitness:
                stagnant = 500
                gb_fitness = copy.deepcopy(particle.fitness)
                gb_position = copy.deepcopy(particle.position)

            particle.move()
            particle.update_velocity(gb_position)

        # check the particles current stagnation value
        stagnant -= 1
        if stagnant <= 0:
            print("PSO terminated early after {} iterations as stagnation was achieved".format(i))
            break

    # for each cluster to be returned, as specified by the parameter
    clusters = [[] for _ in range(number_clusters)]
    # place each point in a cluster
    for point in data:
        fitnesses = []
        # calculate the difference from the point to each centroid we have in our position
        for i in range(int(len(gb_position) / len(point))):
            fitnesses.append(math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, gb_position[i * len(point):(i + 1) * len(point)])])))
        # add the point to the closest cluster
        clusters[fitnesses.index(min(fitnesses))].append(point)

    return clusters

