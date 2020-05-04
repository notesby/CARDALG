import random
import helpers
import numpy as np
import pandas as pd
from copy import copy
from genetic import Experiment,Population,Chromosome,Gen
from joblib import Parallel, delayed,parallel_backend
import time

class GGA(Experiment):
    def __init__(self,data,target,name=None,path=None,verbose=0, mutationRate = 0.05,noChromosomes = 50,noIterations=100,paddingType =1,paddingValue = -1, noChilds = 2,tournaments=2):
        self.data = data
        self.target = target
        self.mutationRate = mutationRate
        self.noChromosomes = noChromosomes
        self.noIterations = noIterations
        self.paddingType = paddingType
        self.paddingValue = paddingValue
        self.noChilds = noChilds
        self.k = tournaments
        self.goals = []
        self.conflicts = []
        self.verbose = verbose
        self.setup()
        Experiment.__init__(self,name,path)
        self.world.verbose = verbose
        
        
    def setup(self):
        if self.verbose >= 1:
            start = time.time()
            print("start setup")
        self.problem,self.conflicts = self.getProblem(self.data,self.target)
        if self.verbose >= 2: print(time.time()-start)
        self.matrix = self.getMatrix(self.problem)
        if self.verbose >= 2: print(np.shape(self.matrix))
        if self.verbose >= 2: print(time.time()-start)
        self.summarized = helpers.summarize(self.matrix,self.paddingValue)
        if self.verbose >= 2: print(time.time()-start)
        self.summarized2 = helpers.summarize2(self.matrix)
        if self.verbose >= 2: print(time.time()-start)
        self.domain = self.getDomain()
        self.goals = list(self.problem.keys())
        if self.verbose >= 1: print(f"{len(self.goals)} goals")
        if self.verbose >= 1: print("end setup")
        
    def getProblem(self,data,target):
        problem = {}
        conflicts = {}
        for row in data:
            cls = row[target]
            values = row[:-1]
            if cls not in problem:
                problem[cls] = {}
            key = str(values)
            conflict = False
            conflictedKeys = []
            for c in problem:
                if c != cls and key in problem[c]:
                    conflict = True
                    conflictedKeys.append(c)
            if not conflict:
                if key not in problem[cls]:
                    problem[cls][key] = [values,0]
                problem[cls][key][1] += 1
            else:
                conflicts[key] = conflictedKeys
        return problem,conflicts
    
    def getMatrix(self,dic):
        matrix = []
        for key in dic:
            for key2 in dic[key]:
                temp = copy(dic[key][key2])
                if type(temp) == np.ndarray:
                    temp = np.append(temp,key)
                else:
                    temp.append(key)
                matrix.append(temp)
        return matrix
    
    def getDomain(self):
        domain = {}
        for row in self.matrix:
            for j,col in enumerate(row[0]):
                if j not in domain:
                    domain[j] = []
                if col not in domain[j]:
                    domain[j].append(col)
        for key in domain:
            #print(domain[key])
            domain[key] = sorted(domain[key])
        return domain
    
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
        keys = list(self.domain.keys())
        keys = sorted(keys)
        for key in keys:
            if len(self.domain[key]) > 10:
                noChoices = random.randint(1,10)
            elif len(self.domain[key]) > 5:
                noChoices = random.randint(1,len(self.domain[key])//2)
            else:
                noChoices = random.randint(1,len(self.domain[key]))
            vals = set(random.choices(self.domain[key],k=noChoices))
            negate = random.randint(0,1)
            chromosome.genes.append(Gen(key,[vals,negate]))
            
    def express(self,chromosome):
        rule = "lambda summary2,goal: ["
        pos = []
        neg = []
        posMatch = []
        negMatch = []
        notZero = 0
        for gen in chromosome.genes:
            temp2 = []
            temp3 = []
            isZero = True
            for val in gen.value[0]:
                if val != -1:
                    isZero=False
                    temp = f"summary2[{gen.identifier}][{val}][goal] if goal in summary2[{gen.identifier}][{val}] else set()"
                    temp2.append(temp)
                    for goal in self.goals:
                        if goal != -1:
                            temp3.append(f"summary2[{gen.identifier}][{val}][{goal}] if goal != {goal} and {goal} in summary2[{gen.identifier}][{val}] else set()")
            notZero += 1 if not isZero else 0
            pos.append(f"set([]).union(*[{','.join(temp2)}])")
            neg.append(f"set([]).union(*[{','.join(temp3)}])")
        rule += f"[{','.join(pos)}],[{','.join(neg)}],{notZero if notZero > 0 else 100000}"+"]"
        return rule
    
    def getTotalMatch(self,phenotype,goal):
        totalNeg = 0
        totalPos = 0
        matchProportion = eval(phenotype)
        totalPosMatch,totalNegMatch,noChromosomes = matchProportion(self.summarized2,goal)
        countPosMatch = 0
        for row in totalPosMatch:
            for col in row:
                countPosMatch += self.matrix[col][1]
        countPosMatch = countPosMatch / noChromosomes
        countNegMatch = 0
        for row in totalNegMatch:
            for col in row:
                countNegMatch += self.matrix[col][1]
        countNegMatch = countNegMatch / noChromosomes
        totalPos = self.summarized["total"][goal]
        totalNeg = self.summarized["total"]["total"]-self.summarized["total"][goal]
        totalExamples = totalNeg+totalPos
        posProp = (totalPos/totalExamples) 
        negProp = (totalNeg/totalExamples) 
        total = (countPosMatch * negProp) + (-1*countNegMatch * posProp)
        maxTotal = (totalPos * negProp)
        res = total/maxTotal
        return res
                    
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
    
    def mutate(self,chromosome):
        if random.random() < self.mutationRate:
            index = random.randint(0,len(chromosome.genes)-1)
            identifier = chromosome.genes[index].identifier
            proportion =  len(self.domain[identifier]) - len(chromosome.genes[index].value)
            if proportion < random.random():
                val = random.choice(self.domain[identifier])
                chromosome.genes[index].value.add(val)
            else:
                if len(chromosome.genes[index].value) > 0:
                    el = random.choice(list(chromosome.genes[index].value))
                    chromosome.genes[index].value.remove(el)
    
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
    
    
    def getRules(self):
        rules = {}
        for population in self.world.populations:
            goal = population.goal
            rules[goal] = None
            bestChromosomes = []+[population.chromosomes[0]]
            terms = {}
            for chromosome in bestChromosomes:
                for gen in chromosome.genes:
                    if gen.identifier not in terms:
                        terms[gen.identifier] = set()
                    if type(gen.value) == set:
                        vals = gen.value.difference(set([-1]))
                        terms[gen.identifier] = terms[gen.identifier].union(vals)
                    else:
                        terms[gen.identifier].add(gen.value)
            lmstr = "lambda x: "
            temp = []
            for key in terms:
                if type(terms[key]) == set:
                    if len(terms[key]) > 0:
                        temp.append(f"(x[{key}] in {terms[key]} if x[{key}] != -1 else True)")
            lmstr += " and ".join(temp)
            fun = eval(lmstr)
            rules[goal]= fun
        return rules
    
    def predict(self,values):
        for key in self.rules:
            if self.rules[key](values):
                return key
        return -1
    
    
    def run(self):
        super().run()
        fileName = "{}.json".format(0)
        self.saveState(fileName)
        for i in range(1,self.noIterations):
            print(f"iteration {i}")
            fileName = "{}.json".format(i)
            self.world.nextGeneration()
            self.saveState(fileName)
        self.rules = self.getRules()
        
    def toDict(self):
        temp = {}
        temp["paddingType"] = self.paddingType
        temp["paddingValue"] = self.paddingValue
        temp["noChromosomes"] = self.noChromosomes
        temp["noChilds"] = self.noChilds
        temp["mutationRate"] = self.mutationRate
        return temp