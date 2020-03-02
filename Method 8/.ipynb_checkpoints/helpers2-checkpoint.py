import numpy as np
import sys
from copy import copy
import math


def getLearningProblem(data,target,removeConflicts=True):
        problem = {}
        conflicts = {}
        for row in data:
            cls = row[target]
            values = row[:-1]
            if cls not in problem:
                problem[cls] = {}
            key = str(values)
            conflict = False
            conflictedKeys = None
            if removeConflicts:
                conflictedKeys = []
                for c in problem:
                    if c != cls and key in problem[c]:
                        conflict = True
                        conflictedKeys.append((cls,c))
            if not conflict:
                if key not in problem[cls]:
                    problem[cls][key] = [values,0]
                problem[cls][key][1] += 1
            else:
                conflicts[key] = conflictedKeys
        return problem,conflicts

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

def getExpr(domains,target):
    fun = f"lambda " + ",".join([f"x{i}" for i in range(target)]) +":"
    variables = []
    for i in range(target):
        count = 1
        for j in range(i+1,target):
            count = count*len(domains[j])
        variables.append(f"{count}*x{i}")
    fun += "+".join(variables)
    print(fun)
    return eval(fun)

def getDomainExp(target):
    fun = "lambda domain,x:"
    variables = []
    for i in range(target):
        variables.append(f"domain[{i}][x[{i}]]")
    temp = ",".join(variables)
    fun += f"[{temp}]"
    print(fun)
    return eval(fun)

def getBinaryStrings(problem,target,removeConflicts=False):
    res = {}
    domains = {}
    counts = {}
    data2,conflicts = getLearningProblem(problem,target,removeConflicts)
    data3 = dictToMat(data2)
    print("getting domains")
    for row in data3:
        for j,col in enumerate(row[0]):
            if j not in domains:
                domains[j] = set()
            domains[j].add(col)
    domains2 = {}
    for key in domains:
        domains2[key] = {}
        domains[key] = sorted(list(domains[key]))
        for k2 in domains[key]:
            if key not in counts:
                counts[key] = 0
            domains2[key][k2] = counts[key]
            counts[key] += 1
    domains = domains2
    print("get row number function")
    getRow = getExpr(domains,target)
    getVals = getDomainExp(target)
    print("building binary strings")
    mn = sys.maxsize
    mx = -sys.maxsize
    #for key in domains[target]:
    #print(f"string for key{key}")
    
    for row in data3:
        noRow = getRow(*getVals(domains,row[0]))
        key = row[2]
        if key not in res:
            res[key] = set()
        res[key].add(noRow)
        mn = min(mn,noRow)
        mx = max(mx,noRow)
    print(mx,mn)
    bstrs = {"total": np.full((mx+1),0,dtype=int)}
    for k in res:
        print(f"string for key{k}")
        bstrs[k] = np.full((mx+1),0,dtype=int)
        for val in res[k]:
            bstrs[k][val] = 1
            bstrs["total"][val] = 1
    return bstrs,mn,mx,res,conflicts

def getCorrelations2(data):
    if len(data.shape) == 2:
        mat = {}
        for row in data:
            mid = math.floor((len(row)-1)/2)
            for j,col in enumerate(row):
                key = row[mid]
                if j != mid and j!=len(row)-1:
                    if key not in mat:
                        mat[key] = [{} for i in range(len(row)-1)]
                    if col not in mat[key][j]:
                        mat[key][j][col] = 0
                    mat[key][j][col] += 1
        return mat
    elif len(data.shape) == 3:
        for state in data:
            pass
    elif len(data.shape) == 4:
        pass
        

def getCorrelations(data,target):
    res = {}
    for row in data:
        for j,col in enumerate(row):
            if j != target:
                key = row[target]
                if key not in res:
                    res[key] = [{} for i in range(len(row)-1)]
                if col not in res[key][j]:
                    res[key][j][col] = 0
                res[key][j][col] += 1
    return res

def getMatrixCorrelations(correlations):
    res = {}
    for key in correlations:
        temp = correlations[key]
        keys = set()
        for j,col in enumerate(temp):
            keys = keys.union( set(col.keys()))
        orderKeys = sorted(list(keys))
        res[key] = {"mat":np.zeros(shape=(len(keys),len(temp))),"rows":orderKeys,"cols":list(range(len(temp)))}
        for i in range(len(orderKeys)):
            for j,col in enumerate(temp):
                k = orderKeys[i]
                if k in col:
                    res[key]["mat"][i][j] = col[k]
    return res
        

def getTable(noVariables):
    noRows=pow(2,noVariables)
    table = np.zeros(shape=(noRows,noVariables),dtype=int)
    print(noRows)
    for j in range(noVariables):
        temp = noRows // pow(2,j+1)
        flip = 0
        counter = 0
        for i in range(noRows):
            table[i,j] = flip
            if flip == 1:
                counter -= 1
                if counter <= 0:
                    flip = 0
            else:
                counter += 1
                if counter >= temp:
                    flip = 1
    return table

from itertools import combinations


def place_ones(size, count):
    for positions in combinations(range(size), count):
        p = [0] * size

        for i in positions:
            p[i] = 1

        yield p

def getFunctions(table):
    noFunctions = pow(2,table.shape[0])
    print(noFunctions)
    functions = list()
    for i in range(table.shape[0]+1):
        functions += place_ones(table.shape[0],i)
    functions = np.array(functions,dtype=int)
    return functions
    
def getNeighborhoodCombinations(noNeighbors,noActiveNeighbors):
    return list(place_ones(noNeighbors,noActiveNeighbors))