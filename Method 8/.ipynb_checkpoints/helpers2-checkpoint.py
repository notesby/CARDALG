import numpy as np
import sys
from copy import copy


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