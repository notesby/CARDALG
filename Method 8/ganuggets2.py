import numpy as np
import math
import random
from copy import copy

def encodeKey(index,value):
    return "{},{}".format(index,value)

def decodeKey(key):
    return list(map(lambda x: int(x),key.split(",")))

#returns an expression to get the transformed coordinates 
# from the original dimensions to the 1 dimension flattened data
def getExpr(size):
    val = ""
    lst = []
    if len(size) > 1:
        for i in range(1,len(size)):
            temp = "xi[{}]".format(i-1)
            for j in range(i,len(size)):
                temp += "*{}".format(size[j])
            lst.append(temp)
    else:
        i = 0
    val += "+".join(lst)
    val += "+xi[{}]".format(i)
    return val


#returns an array with the position in the flattened data
#coords is an array with coordinate relative to the cell in the original dimensions
# size = np.shape(data)
def getNeighbors(cell,coords,size):
    newCoords = []
    expr = getExpr(size)
    for coord in coords:
        xi = []
        outOfBounds = False
        for i,c in enumerate(cell):
            if type(coord) != int:
                v = c+coord[i]
                if v >= size[i] or v < 0:
                    outOfBounds = True
                else:
                    xi.append(v)
            else:
                v = c+coord
                if v >= size[0] or v < 0:
                    outOfBounds = True
                else:
                    xi.append(c+coord)
        if outOfBounds:
            newCoords.append(-1)
        else:
            newCoord = eval(expr)
            newCoords.append(newCoord)
            
    return newCoords

#returns the values of the neighbors of a certain cell
#data = flattened array of the data
#neighbors = the positions of neighbors of a certain cell
#paddingtype = 0 => don't get values,1=> fill with padding value, 2 => don't fill and return empty dict
#paddingvalue = the values to fill when the padding type equals 1
def getNeighborsValue(data,neighbors,paddingType = 0,paddingValue=0):
    values = {}
    for i,n in enumerate(neighbors):
        val = None
        if n >= 0 and n < len(data):
            val = data[n]
        else:
            if paddingType == 0: continue
            elif paddingType == 1:
                val = paddingValue
            elif paddingType == 2:
                values = None
                break
        if val != None:
            values[i] = val
    return values

#returns in each iteration an array with the indexes of each dimension
def multiDimensionalGenerator(size):
    counters = np.array([size[i]-1 for i in range(len(size)-1)])
    counters = np.append(counters,size[-1])
    count = len(size)-1
    while (counters[0] >= 0):
        counters[count] -= 1
        yield [int(i) for i in counters]
        if counters[count] <= 0:
            while(counters[count] <= 0 and count > 0):
                count -= 1
            counters[count] -= 1 
            while(count+1 < len(size)):
                if count+1 == len(size)-1:
                    counters[count+1] = size[count+1]
                else:
                    counters[count+1] = size[count+1]-1
                count += 1

def manhattanDistance(arr):
    res = 0
    for i in arr:
        res += abs(i)
    return res
    
def vonNeumann(radious,distance):
    expr = lambda x: manhattanDistance(x) <= distance
    return getNeighborhood(radious,expr)
    
                
def moore(radious):
    expr = lambda x: True
    neighborhood = getNeighborhood(radious,expr)
    return neighborhood

#returns an array with the neighborhood
#expression = function to filter the neighborhood, receives a list of the indexes according to the dimension
#radious = array with the distance from each dimension                
def getNeighborhood(radious,expression):
    neighborhood = []
    spaces = []
    dimensions = len(radious)
    for i in range(dimensions):
        size = radious[i]
        spaces.append(np.arange(-size, size+1, 1))
    mesh = np.meshgrid(*spaces)
    stack = np.stack(mesh,axis=dimensions)
    stackShape = np.shape(stack)[:-1]
    for index in multiDimensionalGenerator(stackShape):
        tIndex = tuple(index)
        if expression(stack[tIndex]):
            neighborhood.append(stack[tIndex])
    for i in range(dimensions-1,-1,-1):
        neighborhood.sort(key = lambda x: x[i])
    return neighborhood

def dictToMat(dic):
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

def getDomain(data):
    #encontramos los valores observados de cada atributo en los subproblemas de aprendizaje
    domain = {}
    for row in data:
        for j,col in enumerate(row):
            if j not in domain:
                domain[j] = []
            if col not in domain[j]:
                domain[j].append(col)
    for key in domain:
        domain[key].sort()
    return domain

def binarize(data,domain):
    #binarizamos los subproblemas de aprendizaje
    binarizedMatrix = []
    for row in data:
        rtemp = []
        for j,col in enumerate(row):
            if j < len(row)-1:
                for val in domain[j]:
                    if col >= val:
                        rtemp.append(1)
                    else:
                        rtemp.append(0)
            else:
                rtemp.append(col)
        binarizedMatrix.append(rtemp)
    return binarizedMatrix

def getLearningProblem(data,neighborhood,paddingType,paddingValue):
    problem = {}
    dataSize = np.shape(data)
    stateSize = dataSize[1:]
    noStates = dataSize[0]
    for iState in range(1,noStates):
        currentState = data[iState-1].reshape(-1)
        for cell in multiDimensionalGenerator(stateSize):
            index = tuple([iState]+cell)
            cls = data[index]
            if cls not in problem:
                problem[cls] = {}
            neighbors = getNeighbors(cell,neighborhood,stateSize)
            values = getNeighborsValue(currentState,neighbors,paddingType,paddingValue)
            if values != None:
                values = [values[key] for key in values]
                problem[cls][str(values)] = values
    return problem
#individual format (ant,cons) where:
# ant = [[attrInd,val],...,[attrInd,val]]
# cons = [attrInd,val]

#Returns the number of active attributes in the antecedent
def getNumberOfAttributes(ant):
    count = 0;
    for attr in ant:
        if attr[1] != -1:
            count += 1
    return count

#Returns the interestingess degree 
#totalInfoGain = the summatory of the infoGain for each attribute in the antecedent
#noOfAttr = the number of attributes in the antecedent
#domainCardinality = the Cardinality of the goal attribute
def antInterestignessDegree(totalInfoGain,noOfAttr,domainCardinality):
    if (noOfAttr == 0 ):
        return 1
    return 1 - ((totalInfoGain/noOfAttr)/math.log2(domainCardinality))


def consInterestignessDegree(consAttr,noEvents,beta):
    noFavEvents = noEvents[consAttr[0]][consAttr[0]][consAttr[1]][consAttr[1]]
    totalNoEvents = noEvents["totalNoEvents"]
    return math.pow( 1 -  probability(noFavEvents,totalNoEvents),(1/beta) )

#returns the infoGain of an antecedent attribute with a given goal attribute
#attAnt = the antecedent attribute (pair of attr index and value)
#attCons = the consequent attribute (pair of attr index and value)
#domain = the domain of the attributes
#noEvents = tha total number of events for the probability calculation
def infoGain(attAnt,attCons,domain,noEvents):
    #print("---",info(domain,noEvents,attCons),info(domain,noEvents,attCons,attAnt),"----")
    return info(domain,noEvents,attCons) - info(domain,noEvents,attCons,attAnt)

#returns the entropy of the goal attribute or the entropy ot he goal attribute given antecedent attribute
#domain = the domain of the attributes
#noEvents = tha total number of events for the probability calculation
#attCons = the consequent attribute (pair of attr index and value)
#attAnt = the antecedent attribute (pair of attr index and value)
def info(domain,noEvents,attCons,attAnt = None):
    res = 0
    if attAnt == None:
        for val in domain[attCons[0]]:
            noFavEvents = noEvents[attCons[0]][attCons[0]][val][val]
            totalNoEvents = noEvents["totalNoEvents"]
            pr = probability(noFavEvents,totalNoEvents)
            res += (pr*math.log2(pr))
        res = res * -1
    else:
        for val in domain[attAnt[0]]:
            totalNoEvents = noEvents["totalNoEvents"]
            noFavEvents = 0
            for gAttr in noEvents[attAnt[0]]:
                for gVal in noEvents[attAnt[0]][gAttr]:
                    noFavEvents += noEvents[attAnt[0]][gAttr][gVal][val]
            prAntAtt = probability(noFavEvents,totalNoEvents)
            sumCondInfo = 0
            for cVal in domain[attCons[0]]:
                probCA = probability(noEvents[attAnt[0]][attCons[0]][cVal][val],totalNoEvents)
                probA = probability(noFavEvents,totalNoEvents)
                condProb = probCA / probA
                if (condProb>0):
                    sumCondInfo += (condProb*math.log2(condProb))
            sumCondInfo *= -1
            res += sumCondInfo * prAntAtt
    return res
            

def probability(noFavEvents,noEvents):
    return noFavEvents/noEvents

#Calculate the number of events given each possible value of the goal attributes indexes specified
#goalAttributes = an array with the goal attributes
#domain = the domain of the attributes
#dataset = the dataset where the data that will be processed
def calculateNoEvents(goalAttributes,domain,dataset):
    noEventsC = {}
    noEvents = 1
    #for step in dataset:
    #    noEvents += len(step)
    for val in np.shape(dataset)[:-1]:
        noEvents = noEvents*val
    
    for attr in domain:
        noEventsC[attr] = {}
        for g in goalAttributes:
            noEventsC[attr][g] = {}
            for gval in domain[g]:
                noEventsC[attr][g][gval] = {}
                for val in domain[attr]:
                    noEventsC[attr][g][gval][val] = 0
    
    size = np.shape(dataset)
    for index in multiDimensionalGenerator(size):
        ind = tuple(index)
        val = dataset[ind]
        attr = index[-1]
        for g in goalAttributes:
            ind2 = tuple(index[:-1]+[g])
            gval = dataset[ind2]
            noEventsC[attr][g][gval][val] += 1
    noEventsC["totalNoEvents"] = noEvents
    return noEventsC
        
#Returns the accuracy of the antecedent with the consequent
#ant = the array of attributes
#cons =  the attribute
#dataset = the data that will be processed
def predictionAccuracy(ant,cons,dataset):
    acCount = {}
    aCount = 0
    size = np.shape(dataset)[:-1]
    for index in multiDimensionalGenerator(size):
        ind = tuple(index)
        vAnt = True
        row = dataset[ind]
        for att in ant:
            vAnt = vAnt and ((row[att[0]] == att[1]) if att[1] != -1 else True)
        if row[cons[0]] not in acCount:
            acCount[row[cons[0]]] = 0
        if vAnt:
            acCount[row[cons[0]]] += 1
            aCount += 1
    for key in acCount:
        if aCount > 0:
            acCount[key] = (acCount[key] - 1/2)/aCount
    return acCount

def predictionAccuracy2(ant,cons,dataset):
    acCount = {"accepted":{},"rejected":{}}
    aCount = 0
    size = np.shape(dataset)[:-1]
    for index in multiDimensionalGenerator(size):
        ind = tuple(index)
        vAnt = True
        row = dataset[ind]
        for att in ant:
            vAnt = vAnt and ((row[att[0]] == att[1]) if att[1] != -1 else True)
        if row[cons[0]] not in acCount["accepted"]:
            acCount["accepted"][row[cons[0]]] = 0
        if row[cons[0]] not in acCount["rejected"]:
            acCount["rejected"][row[cons[0]]] = 0
        if vAnt:
            acCount["accepted"][row[cons[0]]] += 1
            aCount += 1
        else:
            acCount["rejected"][row[cons[0]]] += 1
    return acCount

def f1score(acc):
    recall = {}
    precision = {}
    f1 = {}
    for key in acc["accepted"]:
        recall[key] = acc["accepted"][key]
        precision[key] = acc["accepted"][key]
        f1[key] = 0
        for key2 in acc["rejected"]:
            if key == key2:
                recall[key] += acc["rejected"][key]
        for key2 in acc["accepted"]:
            if key != key2:
                precision[key] += acc["accepted"][key2]
        recall[key] = acc["accepted"][key] / recall[key]
        precision[key] = (acc["accepted"][key] / precision[key]) if precision[key] != 0 else 0
        if (precision[key] + recall[key]) != 0:
            f1[key] = recall[key] * precision[key] / (precision[key] + recall[key])
        else:
            f1[key] = 0
    return f1
    
#Returns the fitnes of an individual
def gafitness(w1,w2,beta,ant,cons,domain,noEvents,dataset):
    bestGoalValue = 0
    noAttr = 0
    noAttr = getNumberOfAttributes(ant)
    consInt = {}
    sumInfoGain= {}
    antInt = {}
    acc = predictionAccuracy(ant,cons,dataset)
    for val in domain[cons[0]]:
        consInt[val] = consInterestignessDegree([cons[0],val],noEvents,beta)
        if val not in sumInfoGain:
            sumInfoGain[val] = 0
        for attr in ant:
            if attr[1] != -1:
                sumInfoGain[val] += infoGain(attr,[cons[0],val],domain,noEvents)
        antInt[val] = antInterestignessDegree(sumInfoGain[val],noAttr,len(domain[cons[0]]))
        fit = ((w1*(antInt[val] + consInt[val]) / 2) + (w2 * acc[val])) / (w1 + w2)
        #print("fit {},antInt {},consInt {},acc {}".format(fit,antInt[val],consInt[val],acc[val]))
        #print(fit)
        
        if fit > bestGoalValue:
            bestGoalValue = fit
            cons[1] = val
    return bestGoalValue

#Returns the fitnes of an individual
def gafitness2(w1,w2,beta,ant,cons,domain,noEvents,dataset):
    bestGoalValue = 0
    noAttr = 0
    noAttr = getNumberOfAttributes(ant)
    consInt = {}
    sumInfoGain= {}
    antInt = {}
    acc = predictionAccuracy2(ant,cons,dataset)
    acc = f1score(acc)
    #print(acc)
    for val in domain[cons[0]]:
        consInt[val] = consInterestignessDegree([cons[0],val],noEvents,beta)
        if val not in sumInfoGain:
            sumInfoGain[val] = 0
        for attr in ant:
            if attr[1] != -1:
                sumInfoGain[val] += infoGain(attr,[cons[0],val],domain,noEvents)
        antInt[val] = antInterestignessDegree(sumInfoGain[val],noAttr,len(domain[cons[0]]))
        fit = ((w1*(antInt[val] + consInt[val]) / 2) + (w2 * acc[val])) / (w1 + w2)
        #print("fit {},antInt {},consInt {},acc {}".format(fit,antInt[val],consInt[val],acc[val]))
        #print(fit)
        
        if fit > bestGoalValue:
            bestGoalValue = fit
            cons[1] = val
    return bestGoalValue

def initialize(populationSize,antMinSize,antMaxSize,objAttrInd,domain,seed=-1):
    population = []
    if seed != -1:
        random.seed(seed)
    for i in range(populationSize):
        antSize = random.randint(antMinSize,antMaxSize)
        ant = [[i,-1] for i in range(len(domain))]
        for j in range(antSize):
            attr = random.randint(0,len(domain)-1)
            val = random.randint(-1,max(domain[attr]))
            ant[attr][1]= val
        valC = random.randint(min(domain[objAttrInd]),max(domain[objAttrInd]))
        cons = [objAttrInd,valC]
        population.append([ant,cons])
    return population

def countActiveGenes(ant):
    count = 0
    for gen in ant:
        if gen[1] != -1:
            count += 1
    return count

def insertCondition(ant,antMaxSize,domain):
    active = countActiveGenes(ant)
    prob = 1-(active/antMaxSize)
    if random.random() < prob:
        for gen in ant:
            if random.random() < .2 and active < antMaxSize:
                if gen[1] == -1:
                    ind = random.randint(0,len(domain[gen[0]])-1)
                    gen[1] = domain[gen[0]][ind]
                    active += 1
                    prob = 1-(active/antMaxSize)

def removeCondition(ant,antMaxSize,domain):
    active = countActiveGenes(ant)
    prob = (active/antMaxSize)
    if random.random() < prob: 
        for gen in ant:
            if active > 1:
                if random.random() < .2:
                    if gen[1] != -1:
                        gen[1] = -1
                        active -= 1
                        prob = (active/antMaxSize)

def tournament(fitnessTbl,k):
    best = None
    for i in range(k+1):
        ind = random.randint(0, len(fitnessTbl)-1)
        if (best == None) or fitnessTbl[ind][1] > fitnessTbl[best][1]:
            best = ind
    return fitnessTbl[best]

def crossover(parents,population,crossprob):
    offsprings = []
    for i in range(1,len(parents),2):
        p1 = population[parents[i-1][0]][0]
        p2 = population[parents[i][0]][0]
        child1 = [[],[population[parents[i-1][0]][1][0],population[parents[i-1][0]][1][1]]]
        child2 = [[],[population[parents[i][0]][1][0],population[parents[i][0]][1][1]]]
        for j in range(len(p1)):
            if random.random() < crossprob:
                child1[0].append([p2[j][0],p2[j][1]])
                child2[0].append([p1[j][0],p1[j][1]])
            else:
                child1[0].append([p1[j][0],p1[j][1]])
                child2[0].append([p2[j][0],p2[j][1]])
        offsprings.append(child1)
        offsprings.append(child2)
    return offsprings

def mutate(ant,domain,mutationRate):
    for gen in ant:
        if random.random() <= mutationRate:
            ind = random.randint(0,len(domain[gen[0]])-1)
            gen[1] = domain[gen[0]][ind]

def removePopulation(population,fitnessTbl,populationSize):
    newPopulation = []
    newPopulation = [population[fitnessTbl[0][0]] ]#for x in fitnessTbl[:populationSize]]
    for i in range(populationSize):
        elite = tournament(fitnessTbl,2)
        newPopulation.append(population[elite[0]])
        fitnessTbl.remove(elite)
        if len(fitnessTbl) <= 0: break
    return newPopulation
    
def ganuggets(populationSize,noOffsprings,antMinSize,antMaxSize,beta,w1,w2,mutationRate,crossprob,dataset,domain,goalAttr,noEvents,seed,maxIter = 0):
    population = initialize(populationSize,antMinSize,antMaxSize,goalAttr,domain,seed)
    fitnessTbl = []
    for i in range(len(population)):
        fit = gafitness(w1,w2,beta,population[i][0],population[i][1],domain,noEvents,dataset)
        fitnessTbl.append([i,fit,population[i][1][1]])
    it = 0
    fitGoalReached = False
    fitnessHistory = {}
    while it < maxIter and not fitGoalReached:
        print(it)
        it += 1
        fitnessTbl = sorted(fitnessTbl,key = lambda x: x[1],reverse = True)
        
        #select individuals based on fitness
        groupedFitness = {}
        for fit in fitnessTbl:
            if fit[2] not in groupedFitness:
                groupedFitness[fit[2]] = []
            if fit[2] not in fitnessHistory:
                fitnessHistory[fit[2]] = []
            groupedFitness[fit[2]].append(fit)
        parents = {}
        offsprings = []
        for key in groupedFitness:
            if len(groupedFitness[key]) > 0:
                #print("1.- ",groupedFitness[key][0])
                fitnessHistory[key].append(groupedFitness[key][0][1])
                if key not in parents:
                    parents[key] = []
                for i in range(noOffsprings*2):
                    best = tournament(groupedFitness[key],2)
                    parents[key].append(best)
                offsprings += crossover(parents[key],population,crossprob)
                
        for child in offsprings:
            mutate(child[0],domain,mutationRate)
            insertCondition(child[0],antMaxSize,domain)
            removeCondition(child[0],antMaxSize,domain)
        population = population+offsprings
        fitnessTbl = []
        for i in range(len(population)):
            fit = gafitness(w1,w2,beta,population[i][0],population[i][1],domain,noEvents,dataset)
            fitnessTbl.append([i,fit,population[i][1][1]])
        fitnessTbl = sorted(fitnessTbl,key = lambda x: x[1],reverse = True)
        groupedFitness = {}
        for fit in fitnessTbl:
            if fit[2] not in groupedFitness:
                groupedFitness[fit[2]] = []
            groupedFitness[fit[2]].append(fit)
        temPop = []
        for key in groupedFitness:
            #print("2.- ",groupedFitness[key][0])
            if len(groupedFitness[key]) > 0:
                temPop += removePopulation(population,groupedFitness[key],populationSize)
        population = temPop
        fitnessTbl = []
        for i in range(len(population)):
            fit = gafitness(w1,w2,beta,population[i][0],population[i][1],domain,noEvents,dataset)
            fitnessTbl.append([i,fit,population[i][1][1]])
    fitnessTbl = sorted(fitnessTbl,key = lambda x: x[1],reverse = True)
    return fitnessTbl,population,fitnessHistory

def populationPostprocessing(population):
    rules = {}
    for ind in population:
        if ind[1][1] not in rules:
            rules[ind[1][1]] = []
        for gen in ind[0]:
            if gen[1] != -1:
                rules[ind[1][1]].append(["{},{}".format(gen[0],gen[1])])
    return rules


def binarizedToDomain(rules,domain):
    keys = list(domain.keys())
    oRules = {}
    for cls in rules:
        if cls not in oRules:
            oRules[cls] = {}
        prop = []
        for clause in rules[cls]:
            expr = []
            for term in clause:
                col = 0
                ind,val = decodeKey(term)
                bottom = 0
                for key in keys:
                    if ind >= bottom and ind < (bottom + len(domain[key])):
                        col = key
                        ind -= bottom
                        break
                    bottom += len(domain[key])
                relational = (">=" if val== 1 else "<")
                t = "A[{}] {} {}".format(col,relational,domain[col][ind])
                expr.append(t)
            prop.append("({})".format(" or ".join(expr)))
        oRules[cls] = " and ".join(prop)
    return oRules