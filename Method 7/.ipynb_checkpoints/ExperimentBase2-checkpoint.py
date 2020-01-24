import random
import helpers
import numpy as np
import pandas as pd
from copy import copy
from genetic import Experiment,Population,Chromosome,Gen
from joblib import Parallel, delayed,parallel_backend
import time

class CAModel():
    def __init__(self,rules,neighborhood,paddingType,paddingValue):
        self.paddingType = 1
        self.paddingValue = -1
        self.rules = rules
        self.neighborhood = neighborhood
        
    def run(self,initialState,noSteps):
        size = np.shape(initialState)
        initialState = initialState.reshape(-1)
        currentStep = copy(initialState)
        for step in range(noSteps):
            nextStep = copy(currentStep)#np.zeros(shape=np.shape(currentStep))
            for cell in helpers.multiDimensionalGenerator(size):
                neighbors = helpers.getNeighbors(cell,self.neighborhood,size)
                A = helpers.getNeighborsValue(currentStep,neighbors,self.paddingType,self.paddingValue)
                for op in self.rules:
                    res = self.rules[op](A)
                    if res:
                        xi = cell
                        expr = helpers.getExpr(size)
                        tmpCell = expr(xi)
                        nextStep[tmpCell] = op
            yield step,nextStep.reshape(size)
            currentStep = nextStep

class Transformer():
    def __init__(self,experiment):
        self.experiment = experiment
        self.lmstr = {}
    
    def getModel(self):
        rules = self.getRules()
        model = CAModel(rules,self.experiment.neighborhood,self.experiment.paddingType,self.experiment.paddingValue)
        return model
    
    def getRules(self):
        rules = {}
        for population in self.experiment.world.populations:
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
                        pValue = f"{self.experiment.paddingValue}"
                        if type(self.experiment.paddingValue) == str:
                            pValue = f"'{self.experiment.paddingValue}'"
                        temp.append(f"(x[{key}] in {terms[key]} if x[{key}] != {pValue} else True)")
            lmstr += " and ".join(temp)
            self.lmstr[goal] = lmstr
            fun = eval(lmstr)
            rules[goal]= fun
        return rules

class ExperimentBaseModel(Experiment):
    def __init__(self,data,name,path,verbose=0):
        self.data = data
        self.radious = 2
        self.neighborhood = helpers.moore([self.radious])
        self.neighborhoodWLevels = helpers.mooreWLevels([self.radious])
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
        
    def setup(self):
        start = time.time()
        print("start setup")
        self.problem = self.getProblem()
        print(time.time()-start)
        self.matrix = self.getMatrix()
        print(np.shape(self.matrix))
        print(time.time()-start)
        self.summarized = helpers.summarize(self.matrix,self.paddingValue)
        print(time.time()-start)
        self.summarized2 = helpers.summarize2(self.matrix)
        print(time.time()-start)
        self.domain = self.getDomain()
        self.goals = list(self.problem.keys())
        print(f"{len(self.goals)} goals")
        print("end setup")
        
    def getProblemSub2(self,iState,cell,stateSize,currentState):
        index = tuple([iState]+cell)
        cls = self.data[index]
        #start = time.time()
        neighbors = helpers.getNeighbors(cell,self.neighborhood,stateSize)
        #print("neighbors time {}".format(time.time()-start))
        values = helpers.getNeighborsValue(currentState,neighbors,self.paddingType,self.paddingValue)
        #print("neighbors value time {}".format(time.time()-start))
        values = [values[key] for key in values]
        #print("neighbors value 2 time {}".format(time.time()-start))
        return [cls,values]
    
    def getProblemSub1(self,iState,problem,dataSize,stateSize,noStates):
        currentState = self.data[iState-1].reshape(-1)
        start = time.time()
        print("Start {} {}".format(iState,stateSize))
        backend = 'threading'
        with parallel_backend(backend):
            vlscls = Parallel(n_jobs=4)(delayed(self.getProblemSub2) (iState,cell,stateSize,currentState) for cell in helpers.multiDimensionalGenerator(stateSize))
            print("middle {}".format((time.time()-start)))
            for el in vlscls:
                cls = el[0]
                values = el[1]
                if cls not in problem:
                    problem[cls] = {}
                strvalues = str(values)
                if strvalues not in problem[cls]:
                    problem[cls][strvalues] = [None,0]
                problem[cls][strvalues][0] = values
                problem[cls][strvalues][1] += 1
        print("End {} {}".format( iState,(time.time()-start)))
    
    def getProblem(self):
        problem = {}
        dataSize = np.shape(self.data)
        stateSize = dataSize[1:]
        noStates = dataSize[0]
        backend = 'threading'
        with parallel_backend(backend):
            Parallel(n_jobs=4)(delayed(self.getProblemSub1)(iState,problem,dataSize,stateSize,noStates) for iState in range(1,noStates))
        return problem
        
    def getMatrixSub1(self,dic,key):
        backend = 'threading'
        with parallel_backend(backend):
            vals = Parallel(n_jobs=4)(delayed(self.getMatrixSub2)(dic,key,key2) for key2 in dic[key])
        return vals
        
    def getMatrixSub2(self,dic,key,key2):
        #matrix = []
        temp = copy(dic[key][key2])
        if type(temp) == np.ndarray:
            temp = np.append(temp,key)
        else:
            temp.append(key)
        #matrix.append(temp)
        return temp
        
        
    def getMatrix(self):
        matrix = []
        dic = self.problem
        backend = 'threading'
        with parallel_backend(backend):
            vals = Parallel(n_jobs=4)(delayed(self.getMatrixSub1)(dic,key) for key in dic)
        for val in vals:
            matrix += val
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
            chromosome.genes.append(Gen(key,random.choice(self.domain[key])))
    
    def getTotalMatch(self,phenotype,goal):
        total = 0
        matchProportion = eval(phenotype)
        for row in self.matrix:
            total+= matchProportion(row[0]) * row[1]
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
        chromosome = Chromosome(setup,self.express,crossover,self.mutate,population.generation)
        chromosome.parents = [parent1,parent2]
        iGenes = list(self.domain.keys())
        random.shuffle(iGenes)
        mid = len(iGenes)//2
        genes = [Gen(parent1.genes[x].identifier,parent1.genes[x].value) for x in iGenes[:mid]]
        genes += [Gen(parent2.genes[x].identifier,parent2.genes[x].value) for x in iGenes[mid:]]
        chromosome.genes = sorted(genes,key=lambda x: x.identifier)
        return [chromosome]
    
    def mutate(self,chromosome):
        if random.random() < self.mutationRate:
            index = random.randint(0,len(chromosome.genes)-1)
            identifier = chromosome.genes[index].identifier
            chromosome.genes[index].value = random.choice(self.domain[identifier])
    
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
    
    def express(self,chromosome):
        rule = "lambda x: sum(["
        terms = []
        for gen in chromosome.genes:
            if gen.value != -1:
                terms.append("x[{}] == {}".format(gen.identifier,gen.value))
        if (len(terms) > 0):
            rule += " , ".join(terms) + "]) / {}".format(len(chromosome.genes))
        else:
            rule = "lambda x: 0"
        return rule
    
    def run(self):
        super().run()
        fileName = "{}.json".format(0)
        self.saveState(fileName)
        for i in range(1,self.noIterations):
            print(f"iteration {i}")
            fileName = "{}.json".format(i)
            self.world.nextGeneration()
            self.saveState(fileName)
        
    def toDict(self):
        temp = {}
        temp["data"] = self.data.tolist()
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