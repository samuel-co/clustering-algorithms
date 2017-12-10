

import numpy
import random
import numpy.random as nrand

class Ant:

    def __init__(self, y, x, grid):
        self.location = numpy.array([y, x])
        self.carrying = grid.get_grid()[y][x]
        self.grid = grid


    def decide_action(self, data, y, x, neighbor, conv):
        dy = y - neighbor
        dx = x - neighbor

        # chekcing neighbors
        for i in range((neighbor * 2) + 1):
            xi = (dx + i) % self.dim[0]
            for j in range((neighbor * 2) + 1):
                if j != x and i != y:
                    yj = (dy + j) % self.dim[1]
                    # get neighbor
                    o = self.grid[xi][yj]
                    # how to check similarity???


    def pickup(self, neighbor, conv):
        ant = self.grid.get_grid()[self.loc[0]][self.loc[1]]
        return 1 - self.grid.decide_action(ant, self.loc[0], self.loc[1], neighbor, conv)


    def drop(self, neighbor, conv):
        ant = self.carrying
        return self.grid.decide_action(ant, self.loc[0], self.loc[1], neighbor, conv)


    def move(self, neighbor, conv):
        step = random.randint(1, 25)
        self.location += nrand.randint(-1 * step, 1 * step, 2)
        # mod location to prevent overflow
        location = numpy.mod(self.loc, self.grid.dim)
        # get object at location
        obj = self.grid.get_grid()[location[0]][location[1]]
        # check cell and move if occupied
        if obj is not None:
            # not carrying object
            if self.carrying is None:
                # check to pickup
                if self.pickup(neighbor, conv) >= random.random():
                    # pickup
                    self.carrying = obj
                    self.grid.get_grid()[location[0]][location[1]] = None
                # if not move
                else:
                    self.move(neighbor, conv)
            # move if carrying an object
            else:
                self.move(neighbor, conv)
        # empty cell
        else:
            if self.carrying is not None:
                # Check if the ant drops the object
                if self.drop(neighbor, conv) >= random.random():
                    # Drop the object at the empty location
                    self.grid.get_grid()[location[0]][location[1]] = self.carrying
                    carrying = None


    def evaluate_fitness(self):
        pass

def aco_clustering(data, fill, num_ants, iterations):
    height = 10
    width = 10
    grid = numpy.empty((height, width))
    dim = numpy.array([height, width])
    antAgents = []

    #neighborhood
    neighbor = 25
    #convergence
    conv = 10

    #creating grid
    for y in range(dim[0]):
        for x in range(dim[1]):
            if random.random() <= fill:
                r = random.randint(0, 1)
                if r == 0:
                    grid[y][x] = data(random.normal(5, 0.25, 10))
                elif r == 1:
                    grid[y][x] = data(random.normal(-5, 0.25, 10))
    #creating ants
    for i in range(num_ants):
        ant = Ant.ant(random.randint(0, height - 1), random.randint(0, width - 1), grid)
        antAgents.append(ant)
    for i in range(iterations):
        for ant in antAgents:
            ant.move(neighbor, conv)

def find_clusters():
    pass

def get_grid(self):
    return self.grid
