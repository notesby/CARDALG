import random
import helpers
import numpy as np
import pandas as pd
from copy import copy
from copy import deepcopy
from genetic import Experiment,Population,Chromosome,Gen
from joblib import Parallel, delayed,parallel_backend
import time

class ExperimentBaseModel(Experiment):
    def __init__(self,s1,s2,name,path,target,verbose=0):
        self.target = target
        self.s1 = s1
        self.s2 = s2
        self.radious = 2
        self.neighborhood = helpers.moore([self.radious])
        self.paddingType = 1
        self.paddingValue = -1
        self.antMaxSize = len(self.neighborhood)#(self.radious*2)+1
        self.antMinSize = 2
        self.noChromosomes = 20
        self.noChilds = 5
        self.noIterations = 50
        self.k = 2
        self.mutationRate = .1
        self.goals = []
        self.setup()
        Experiment.__init__(self,name,path)
        self.world.verbose = verbose
    
    def getDomain(self,s1,target):
        unique = set()
        for row in s1[target]:
            for col in row:
                unique.add(col)
        return unique
    
    def getDomains(self,s1,target):
        domains = {}
        for i,layer in enumerate(s1):
            if i != target:
                if i not in domains:
                    domains[i] = set()
                for row in layer:
                    for col in row:
                        domains[i].add(col)
        return domains
    
    def getDomainPositions(self,s1,target):
        positions = {}
        for i,row in enumerate(s1[target]):
            for j,col in enumerate(row):
                if col not in positions:
                    positions[col] = []
                positions[col].append((i,j))
        return positions
    
    def getNeighborsValue(self,s1,position,target):
        stateSize = np.shape(s1[target])
        values = []
        for coord in self.neighborhood:
            newCoord = coord+position
            if (newCoord[0] <= 0 or newCoord[0] >= stateSize[0]) or (newCoord[1] <= 0 or newCoord[1] >= stateSize[1]):
                values.append(-1)
            else:
                layer = s1[target]
                i,j = newCoord
                values.append(layer[i,j])
        return values
    
    def setup(self):
        start = time.time()
        print("start setup")
        self.domain = self.getDomains(self.s1,self.target)
        self.goals = self.getDomain(self.s1,self.target)
        self.positions = self.getDomainPositions(self.s1,self.target)
        print(time.time()-start)
        print("end setup")
    
    def setupWorld(self,world):
        for goal in self.goals:
            population = Population(self.setupPopulation,self.selection,self.regulation,self.fitness,goal)
            world.populations.append(population)
    
    def setupPopulation(self,population):
        setup = lambda x: self.setupChromosome(x,population.goal)
        crossover = lambda p1,p2: self.crossover(p1,p2,population)
        for i in range(self.noChromosomes):
            chromosome = Chromosome(setup,self.express,crossover,self.mutate,population.generation)
            population.chromosomes.append(chromosome)
    
    def setupChromosome(self,chromosome,goal):
        pos = random.choice(self.positions[goal])
        values = []
        for layer in self.s1:
            values.append(layer[pos])
        values = values + self.getNeighborsValue(self.s1,pos,self.target)
        for i,val in enumerate(values):
            chromosome.genes.append(Gen(i,set([val])))
        chromosome.genes.append(Gen(len(chromosome.genes),0))
    
    def express(self,chromosome):
        rule = "lambda x: sum(["
        terms = []
        for gen in chromosome.genes[:5]:
            if gen.value != -1:
                terms.append(f"(1 if x[{gen.identifier}] in {gen.value} else 0)")
        if (len(terms) > 0):
            rule += " , ".join(terms) + f"]) / {len(chromosome.genes[:5])}"
        else:
            rule = "lambda x: 0"
        if type(chromosome.genes[-1].value) == set:
            for i in range(len(chromosome.genes)):
                print("error",chromosome.genes[i].value,chromosome.genes[i].identifier)
            print("error",len(chromosome.genes),chromosome.genes[-1].value,chromosome.genes[-1].identifier)
        return rule,chromosome.genes[-1].value
    
    def getTotalMatch(self,phenotype,goal):
        positions = random.choices(self.positions[goal],k=len(self.positions[goal])//2)
        matchProportion = eval(phenotype[0])
        total = 0
        for pos in positions:
            values = []
            for layer in self.s1:
                values.append(layer[pos])
            total += matchProportion(values)
        total -= phenotype[1]
        return total
                    
    def fitness(self,phenotype,goal):
        return self.getTotalMatch(phenotype,goal)
    
    def tournament(self,chromosomes):
        best = None
        for i in range(self.k+1):
            ind = random.randint(0, len(chromosomes)-1)
            if (best == None) or chromosomes[ind] > chromosomes[best]:
                best = ind
        return chromosomes[best]
    
    def selection(self,chromosomes):
        parents = []
        for i in range(self.noChilds):
            parent1 = self.tournament(chromosomes)
            parent2 = self.tournament(chromosomes)
            parents.append([parent1,parent2])
        return parents
    
    def crossover(self,parent1,parent2,population):
        setup = lambda x: self.setupChromosome(x,population.goal)
        crossover = lambda p1,p2: self.crossover(p1,p2,population)
        chromosome1 = Chromosome(setup,self.express,crossover,self.mutate,population.generation)
        chromosome2 = Chromosome(setup,self.express,crossover,self.mutate,population.generation)
        chromosome1.parents = [parent1,parent2]
        chromosome2.parents = [parent1,parent2]
        iGenes = list(range(len(parent1.genes)))#list(self.domain.keys())
        random.shuffle(iGenes)
        mid = len(iGenes)//2
        genes1 = [Gen(parent1.genes[x].identifier,copy(parent1.genes[x].value)) for x in iGenes[:mid]]
        genes1 += [Gen(parent2.genes[x].identifier,copy(parent2.genes[x].value)) for x in iGenes[mid:]]
        chromosome1.genes = sorted(genes1,key=lambda x: x.identifier)
        genes2 = [Gen(parent2.genes[x].identifier,copy(parent2.genes[x].value)) for x in iGenes[:mid]]
        genes2 += [Gen(parent1.genes[x].identifier,copy(parent1.genes[x].value)) for x in iGenes[mid:]]
        chromosome2.genes = sorted(genes2,key=lambda x: x.identifier)
        return [chromosome1,chromosome2]
    
    def mutate(self,chromosome):
        if random.random() < self.mutationRate:
            index = random.randint(0,len(chromosome.genes[:5])-1)
            identifier = chromosome.genes[index].identifier
            proportion =  len(self.domain[identifier]) - len(chromosome.genes[index].value)
            if proportion < random.random():
                val = random.choice(self.domain[identifier])
                chromosome.genes[index].value.add(val)
            else:
                if len(chromosome.genes[index].value) > 0:
                    el = random.choice(list(chromosome.genes[index].value))
                    chromosome.genes[index].value.remove(el)
    
    def regulation(self,population):
        population.chromosomes = sorted(population.chromosomes,key = lambda x:x.fitness,reverse = True)
        mid = self.noChromosomes // 2
        indexes = [x for x in range(1,len(population.chromosomes))]
        random.shuffle(indexes)
        indexes = indexes[:self.noChromosomes]
        firstHalf = indexes[:mid]
        secondHalf = indexes[mid:]
        selected = [population.chromosomes[0]]
        selected += (np.array(population.chromosomes)[firstHalf]).tolist()
        selected += (np.array(population.chromosomes)[secondHalf]).tolist()
        population.chromosomes = selected
    
    
    def getRule(self,population):
        clausules = []
        for chromosome in population.chromosomes:
            terms = []
            for gen in chromosome.genes[:-1]:
                temp = f"(x[{gen.identifier}] in {gen.value})"
                terms.append(temp)
            clausules.append(" and ".join(terms))
        rule = "lambda x: "+" or ".join(terms)
        return eval(rule)
            
    def getWorldFitness(self,s2,rules,populations):
        prev = deepcopy(s2)
        nex = deepcopy(s2)
        for it in range(200):
            for i,row in enumerate(prev):
                for j,col in enumerate(row):
                    pos = (i,j)
                    values = []
                    for l,layer in enumerate(self.s1):
                        print(it,np.shape(layer),l,i,j)
                        values.append(layer[i,j])
                    values = values + self.getNeighborsValue(self.s1,pos,self.target)
                    for key in rules:
                        if rules[key](values):
                            nex[self.target,i,j] = key
            prev = deepcopy(nex)
        errors = {}
        for i,row in enumerate(s2[self.target]):
            for j,col in enumerate(row): 
                if nex[self.target,i,j] != col:
                    if col not in errors:
                        errors[col] = 0
                    errors[col] += 1
                    if nex[self.target,i,j] not in errors:
                        errors[nex[self.target,i,j]] = 0
                    errors[nex[self.target,i,j]] += 1
        for pop in populations:
            for chromosome in pop.chromosomes:
                chromosome.genes[-1].value = errors[pop.goal]
    
    def run(self):
        super().run()
        fileName = "{}.json".format(0)
        self.saveState(fileName)
        for i in range(1,self.noIterations):
            print(f"iteration {i}")
            fileName = "{}.json".format(i)
            self.world.nextGeneration()
            rules = {}
            for population in self.world.populations:
                rules[population.goal]=self.getRule(population)
            self.getWorldFitness(self.s2,rules,self.world.populations)
            self.saveState(fileName)
        
    def toDict(self):
        temp = {}
        #temp["data"] = self.data.tolist()
        temp["radious"] = self.radious
        temp["neighborhood"] = [x.tolist() for x in self.neighborhood]
        temp["paddingType"] = self.paddingType
        temp["paddingValue"] = self.paddingValue
        temp["antMaxSize"] = self.antMaxSize
        temp["antMinSize"] = self.antMinSize
        temp["noChromosomes"] = self.noChromosomes
        temp["noChilds"] = self.noChilds
        temp["k"] = self.k
        temp["mutationRate"] = self.mutationRate
        return temp