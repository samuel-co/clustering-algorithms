import math
import random
import statistics
class particle:
    def __init__(self,location, fitness):
        self.location=location
        self.fitness=fitness
        self.best=location
        self.velocity=[]
        self.cluster=[]
        for i in location:
            self.velocity.append(random.random())
    def new_best(self,fitness):
        self.best=self.location[:]
        self.fitness=fitness
    def new_location(self,lbest):
        for i in range(len(self.velocity)):
            self.velocity[i]=self.velocity[i]+1*random.random()*(lbest[i]-self.location[i])+1*random.random()*(self.best[i]-self.location[i])
            while(math.fabs(self.velocity[i])>200):
                self.velocity[i]=self.velocity[i]/2
        for i in range(len(self.velocity)):
            self.location[i]=self.location[i]+self.velocity[i]
        return self.location

def PSOcluster (data, particle_num, iterations=100): #particle num = max clusters
    clusters=[]
    particles=[]
    gbest=100
    gb=[]
    for i in range(particle_num):
        #initialize particles as a random point in data, then add noise
        temp=random.choice(data)
        particles.append(particle(temp,100))
        particles[-1].new_location(particles[0].location)
        if(particles[-1].fitness<gbest):
            gbest=particles[-1].fitness
            gb=particles[-1].location[:]
    for p in particles:
        print(p.fitness)
    for count in range(iterations):
        #add points to clusters
        make_clusters(particles,data)
        for p in particles:
            #check fitness of clusters (SSE)
            fit, deletes=fitness(p,data)
            for d in deletes:
                data.remove(d)
            if(fit<gbest):
                #update best fit cluster
                gbest=fit
                gb=p.location[:]
            if (fit<p.fitness):
                p.new_best(fit)
            p.new_location(gb)
        print(count,gb,gbest,"\n")
    for p in particles:
        if len(p.cluster)>0:
            clusters.append(p.cluster)
        print(p.fitness)
    return clusters
# def fitness(part, data):
#     '''average distance from centroid to points in cluster'''
#     neighbor_distance = []
#     # for each point in the data, check its distance from the passed in point.
#     for candidate in part.cluster:
#         # if the point is within range, add it to the list of neighbors
#         distance=math.sqrt(sum([(a - b) ** 2 for a, b in zip(candidate[:-1], part.location[:-1])]))
#         neighbor_distance.append(distance)
#     if(len(neighbor_distance)==0):
#         return 100
#     return (sum(neighbor_distance)/len(neighbor_distance))
def fitness(part,data, last=False):
    if(len(part.cluster)==0):
        part=particle(random.choice(data),100)
        return 100,[]
    elif(len(part.cluster)<len(data)/100):
        part=particle(random.choice(data),100)
        return 100,[]
    cluster_sum = 0
    feature_sums = []
    total_sum=0
    SSE_points=[]
    for i in range(len(part.cluster[0])):
        feature_sums += [sum([point[i] for point in part.cluster])]
    mean_point = [feature_sum/len(part.cluster) for feature_sum in feature_sums]

    # sum the SSE of each point in the cluster
    for point in part.cluster:
        point_SSE = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, mean_point)])) ** 2
        total_sum += point_SSE
        SSE_points.append(point_SSE)

    std=statistics.stdev(SSE_points)
    mean=statistics.mean(SSE_points)
    print(len(SSE_points),len(part.cluster))
    deletes=[]
    for i in range(len(SSE_points)):
        if SSE_points[i]>mean+(3*std):
            deletes.append(part.cluster[i])
    return (total_sum,deletes)

def make_clusters(particles,data):
    '''add each data point to the cluster of the closest particle'''
    for p in particles:
        p.cluster.clear()
    for point in data:
        d=1000
        for p in particles:
            distance=math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, p.location)]))
            if distance<d:
                d=distance
                temp=p
        temp.cluster.append(point)
