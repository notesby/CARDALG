import numpy as np
import sys

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

def getBinaryStrings(problem,target):
    res = {}
    domains = {}
    counts = {}
    print("getting domains")
    for row in problem:
        for j,col in enumerate(row):
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
    for key in domains[target]:
        print(f"string for key{key}")
        res[key] = set()
        for row in problem:
            noRow = getRow(*getVals(domains,row[:-1]))
            res[key].add(noRow)
            mn = min(mn,noRow)
            mx = max(mx,noRow)
    print(mx,mn)
    bstrs = {}
    for k in res:
        print(f"string for key{k}")
        bstrs[k] = np.full((mx+1),0,dtype=int)
        for val in res[k]:
            bstrs[k][val] = 1
    return bstrs,mn,mx