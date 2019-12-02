import math
import uuid
import os
import json
from datetime import datetime, date, time
from joblib import Parallel, delayed

class Gen():
    def __init__(self,identifier,value):
        self.identifier = identifier
        self.value = value
    
    def __eq__(self,other):
        if type(other) == Gen:
            return self.identifier == other.identifier and self.value == other.value
        
    def toDict(self):
        temp = {}
        temp["identifier"] = str(self.identifier)
        temp["value"] = str(self.value)
        return temp
        
class Chromosome():
    def __init__(self,setup,express,crossover,mutate,generation):
        self.uuid = str(uuid.uuid4())
        self.parents = []
        self.genes = []
        self.generation = generation
        self.crossover = crossover
        self.mutate = mutate
        self.express = express
        self.fitness = 0
        self.setup = setup
        self.setup(self)
            
    #crossover operator
    def __add__(self,other):
        if type(other) == Chromosome:
            return self.crossover(self,other)
        return None
    
    #mutate operator
    def __invert__(self):
        return self.mutate(self)
    
    #returns the phenotype
    def __neg__(self):
        return self.express(self)
    
    def __lt__(self,other):
        if type(other) == Chromosome:
            return self.fitness < other.fitness
        return False
    
    def __le__(self,other):
        if type(other) == Chromosome:
            return self.fitness <= other.fitness
        return False
    
    def __gt__(self,other):
        if type(other) == Chromosome:
            return self.fitness > other.fitness
        return False
    
    def __ge__(self,other):
        if type(other) == Chromosome:
            return self.fitness >= other.fitness
        return False
    
    def toDict(self):
        temp = {}
        temp["uuid"] = self.uuid
        temp["generation"] = self.generation
        temp["genes"] = [gen.toDict() for gen in self.genes]
        temp["fitness"] = str(self.fitness)
        temp["parents"] = [str(parent.uuid) for parent in self.parents]
        return temp
    
class Population():
    def __init__(self,setup,selection,regulation,fitness,goal):
        self.chromosomes = []
        self.generation = 0
        self.selection = selection
        self.regulation = regulation
        self.fitness = fitness
        self.goal = goal
        self.setup = setup
        self.setup(self)
    
    def parallelFitness(self,chromosome,goal):
        chromosome.fitness = self.fitness( -chromosome,goal)
    
    def nextGeneration(self):
        self.generation += 1
        backend = 'threading'
        Parallel(n_jobs=4,backend=backend)(delayed(self.parallelFitness)(chromosome,self.goal) for chromosome in self.chromosomes)
        #for chromosome in self.chromosomes:
        #    chromosome.fitness = self.fitness( -chromosome, self.goal)
        childs = []
        for parents in self.selection(self.chromosomes):
            if parents == None:
                raise Exception("parents shouldn't be None.")
            if len(parents) != 2:
                raise Exception('parents length should be 2. The length of parents was: {}'.format(len(parents)))
            childs += parents[0] + parents[1] #crossover
        for child in childs:
            ~child #mutation
            child.fitness = self.fitness(-child,self.goal)
            self.chromosomes.append(child) 
        self.regulation(self)
        
    def avg(self):
        res = 0
        if len(self.chromosomes) == 0:
            return -1
        for chromosome in self.chromosomes:
            res += self.fitness( -chromosome, self.goal)
        res = res / len(self.chromosomes)
        return res
    
    def std(self):
        if len(self.chromosomes) == 0:
            return -1
        res = 0
        avg = self.avg()
        for chromosome in self.chromosomes:
            fitness = self.fitness( -chromosome, self.goal)
            res += pow(avg - fitness, 2)
        res = math.sqrt(res/len(self.chromosomes))
        return res
    
    def toDict(self):
        temp = {}
        temp["generation"] = self.generation
        temp["goal"] = str(self.goal)
        temp["chromosomes"] = [chromosome.toDict() for chromosome in self.chromosomes]
        temp["average"] = self.avg()
        temp["std"] = self.std()
        return temp
        
class World():
    def __init__(self,setup):
        self.populations = []
        self.generation = 0
        self.setup = setup
        self.setup(self)
        
    def nextIteration(self,population):
        population.nextGeneration()
        
    def nextGeneration(self):
        self.generation += 1
        backend = 'threading'
        Parallel(n_jobs=4,backend=backend)(delayed(self.nextIteration)(population) for population in self.populations)
        #for population in self.populations:
        #    population.nextGeneration()
        
    def migrate(self,pi1,pi2):
        pass
    
    def toDict(self):
        temp = {}
        temp["generation"] = self.generation
        temp["populations"] = [population.toDict() for population in self.populations]
        return temp
    
class Experiment():
    def __init__(self,name,path):
        self.name = name
        self.path = path
        self.date = ""
        self.world = World(self.setupWorld)
        
    def saveState(self,fileName):
        path = self.path
        if path == None:
            path = "{}/{}".format(self.name,self.date)
        else:
            path = "{}/{}".format(path,self.date)
        if not os.path.exists(path):
            os.makedirs(path)
        path = "{}/{}".format(path,fileName)
        with open(path, 'w') as json_file:
            json.dump(self.world.toDict(), json_file)
        
    def saveConfig(self):
        path = self.path
        if path == None:
            path = "{}/{}".format(self.name,self.date)
        else:
            path = "{}/{}".format(path,self.date)
        if not os.path.exists(path):
            os.makedirs(path)
        path = "{}/{}".format(path,"config.json")
        with open(path, 'w') as json_file:
            json.dump(self.toDict(), json_file)
        
    def run(self):
        dt = datetime.now()
        self.date = dt.strftime("%d_%m_%y_%H_%M_%S")
        self.saveConfig()
        
    def setupWorld(self,world):
        pass
    
    def setupPopulation(self,population):
        pass
    
    def setupChromosome(self,chromosome):
        pass
    
    def express(self,chromosome):
        pass
    
    def fitness(self,chromosome,goal):
        pass
    
    def selection(self,chromosomes):
        pass
    
    def crossover(self,parent1,parent2):
        pass
    
    def mutate(self,chromosome):
        pass
    
    def regulation(self,population):
        pass
    
    def toDict(self):
        pass
        