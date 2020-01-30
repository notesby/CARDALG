import numpy as np
import random
from copy import copy
from joblib import Parallel, delayed,parallel_backend

class OCAT():
    def __init__(self,data,targetColumn):
        self.data = data
        self.target = targetColumn
        self.domain = {}
        self.indexes = {}
        self.auxiliar = {}
        self.rules = []
        self.terms = []
        self.preprocessing()
        
    def checkConflict(self,problem,cls,key):
        if key in problem[cls]:
            return cls
    
    def getLearningProblem(self,data,target):
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
                    conflictedKeys.append((cls,c))
            if not conflict:
                if key not in problem[cls]:
                    problem[cls][key] = [values,0]
                problem[cls][key][1] += 1
            else:
                conflicts[key] = conflictedKeys
        return problem,conflicts
        
    def dictToMat(self,dic):
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

    def getDomain(self,data,target):
        #encontramos los valores observados de cada atributo en los subproblemas de aprendizaje
        domain = {}
        for row in data:
            for j,col in enumerate(row[0]):
                if j not in domain:
                    domain[j] = set()
                if col not in domain[j]:
                    domain[j].add(col)
        domain[target] = set()
        for row in data:    
            domain[target].add(row[2])
        for key in domain:
            domain[key] = sorted(domain[key])
        return domain
    
    def getBinaryDomain(self,binarized):
        cols = len(binarized[0][0])
        domain = {}
        for col in range(cols):
            domain[col] = [0,1]
        return domain

    def binarize(self,data,domain):
        #binarizamos los subproblemas de aprendizaje
        binarizedMatrix = []
        for row in data:
            rtemp = []
            for j,col in enumerate(row[0]):
                for val in domain[j]:
                    if col >= val:
                        rtemp.append(1)
                    else:
                        rtemp.append(0)
            binarizedMatrix.append([rtemp,row[1],row[2]])
        return binarizedMatrix

    def getPosNeg(self,domain,binarized,indexes,target):
        # obtenemos los ejemplos positivos y negativos para el atributo x[1] de los subproblemas
        keys = list(domain.keys())
        pos_neg = {}
        pos = {}
        for val in domain[keys[target]]:
            if val not in pos_neg:
                pos_neg[val] = {"pos":{},"neg":{}}
        for val in indexes:
            for t in indexes[val]["pos"]:
                i= t[0]
                row = binarized[i]
                for j,col in enumerate(row[0]):
                    if j not in pos_neg[val]["pos"]:
                        pos_neg[val]["pos"][j] = {}
                    if col not in pos_neg[val]["pos"][j]:
                        pos_neg[val]["pos"][j][col] = set()
                    pos_neg[val]["pos"][j][col].add((i,row[1]))
            for t in indexes[val]["neg"]:
                i = t[0]
                row = binarized[i]
                for j,col in enumerate(row[0]):
                    if j not in pos_neg[val]["neg"]:
                        pos_neg[val]["neg"][j] = {}
                    if col not in pos_neg[val]["neg"][j]:
                        pos_neg[val]["neg"][j][col] = set()
                    pos_neg[val]["neg"][j][col].add((i,row[1]))
        return pos_neg
    
    def getIndexes(self,data,domain,target):
        indexes = {}
        print("dindex")
        for val in domain[target]:
            indexes[val] = {"pos":set(),"neg":set()}
        print("pos index")
        for i,row in enumerate(data):
            val = row[2]
            indexes[val]["pos"].add((i,row[1]))
        print("neg index")
        for key in indexes:
            neg = set()
            for key2 in indexes:
                if key != key2:
                    neg = neg.union(indexes[key2]["pos"])
            indexes[key]["neg"] = neg
        return indexes
    
    def getTerms(self,domain,target):
        terms = []
        for key in domain:
            if key != target:
                terms.append((key,0))
                terms.append((key,1))
        return terms
        
    def preprocessing(self):
        print("problem")
        problem,self.conflicts = self.getLearningProblem(self.data,self.target)
        print("matrix")
        matrix = self.dictToMat(problem)
        print("domain")
        self.domain = self.getDomain(matrix,self.target)
        print("binarized")
        binarized = self.binarize(matrix,self.domain)
        print("bdomains")
        self.bdomains = self.getBinaryDomain(binarized)
        print("indexes")
        self.indexes = self.getIndexes(binarized,self.domain,self.target)
        print("auxiliar")
        self.auxiliar = self.getPosNeg(self.domain,binarized,self.indexes,self.target)
        print("terms")
        self.terms = self.getTerms(self.bdomains,self.target)
        
    def getFitnessValue(self,term,auxiliar,removedPos,removedNeg):
        temp = term
        pos = temp[0]
        val = temp[1]
        if pos not in auxiliar["pos"] or pos not in auxiliar["neg"]:
            return [term,None]
        if val not in auxiliar["pos"][pos] and val not in auxiliar["neg"][pos]:
            return [term,None]
        if val in auxiliar["pos"][pos]:
            remainingPos = auxiliar["pos"][pos][val].difference(removedPos) 
            if len(remainingPos) > 0:
                posEj = np.array(list(remainingPos)).sum(axis=0)[1] 
            else:
                posEj = 0.000000001
        else:
            return [term,None]
        if val in auxiliar["neg"][pos]:
            remainingNeg = auxiliar["neg"][pos][val].difference(removedNeg)
            
            if len(remainingNeg) > 0:
                negEj =  np.array(list(remainingNeg)).sum(axis=0)[1] #len(remainingNeg) 
            else:
                #print("errror ",temp,auxiliar["neg"][pos][val],removedNeg,remainingNeg,posEj)
                return [term,2*posEj]
            if negEj == 0:
                return [term,2*posEj]
        else:
            return [term,0.000000001]
        return [term,posEj/negEj]
        
    
    def getProbabilities(self,termsFitness):
        total = 0
        for term in termsFitness:
            total += term[1]
        probabilities = []
        cur = 0
        for i,term in enumerate(termsFitness):
            cur += term[1]/total
            probabilities.append([i, cur])
        return probabilities
    
    def createClause(self,auxiliar,element):
        clausule = set()
        for key in auxiliar:
            for j in auxiliar["neg"]:
                for col in auxiliar["neg"][j]:
                    if element in auxiliar["neg"][j][col]:
                        clausule.add((j,1 if col == 0 else 0))
        return clausule
    
    def obtainRules(self,pos,neg,auxiliar,terms):
        removedNeg = set()
        clausules = []
        count = 0
        backend = 'threading'
        with Parallel(n_jobs=4,backend=backend) as parallel:
            while(len(neg.difference(removedNeg))>0):
                if count % 10 == 0:
                    print(len(neg.difference(removedNeg)))
                removedPos = set()
                termsTemp = [term for term in terms]
                clausule = set()
                while(len(pos.difference(removedPos))>0):
                    termsFitness = parallel(delayed(self.getFitnessValue)(term,auxiliar,removedPos,removedNeg) for term in termsTemp)
                    termsFitness = list(filter(lambda x: x[1] != None,termsFitness))
                    termsFitness = sorted(termsFitness,key=lambda x:x[1],reverse=True)
                    probabilities = self.getProbabilities(termsFitness[:len(termsFitness)//2])
                    #print(termsFitness)
                    rand = random.random()
                    selected = 0
                    for prob in probabilities:
                        if rand > prob[1]:
                            selected = prob[0]
                        else: break
                    term = termsFitness[selected][0]
                    clausule.add(term)
                    termsTemp = list(filter(lambda x: x[0] != term[0],termsTemp))
                    if term[1] in auxiliar["pos"][term[0]]:
                        removedPos = removedPos.union(auxiliar["pos"][term[0]][term[1]])
                temp = set().union(neg)
                #print(clausule)
                for item in clausule:
                    val = 1 if item[1] == 0 else 0
                    post = item[0]
                    temp = temp.intersection(auxiliar["neg"][post].get(val,set()))
                #print(neg.difference(removedNeg))
                if len(temp) > 0 and len(temp.difference(removedNeg)):
                    removedNeg = removedNeg.union(temp)
                    clausules.append(clausule)
                    count = 0
                else:
                    count += 1
                    if count% 100 == 0:
                        print(f"difficult element {count}/1000 to delete it")
                    if count >= 1000:
                        count = 0
                        problematicElements = neg.difference(removedNeg)
                        element = random.choice(list(problematicElements))
                        clausules.append(self.createClause(auxiliar,element))
                        removedNeg.add(element)
                        print(element,neg.difference(removedNeg))
        return clausules
                  
        
    def train(self):
        rules = []
        for val in self.indexes:
            print(f"training for value {val}")
            rules.append([val,self.obtainRules(self.indexes[val]["pos"],self.indexes[val]["neg"],self.auxiliar[val],self.terms)])
        self.rules = rules
        self.rules = self.unbinarize(rules)
        self.rules2 = self.getRules()
        return self.rules
        
    def unbinarize(self,rules):
        newRules = []
        ranges = []
        start = 0
        for key in self.domain:
            end = start+len(self.domain[key])-1
            ranges.append([start,end,key])
            start = end+1
        for row in rules:
            newRow = []
            for col in row[1]:
                clause = set()
                for term in col:
                    oCol = 0
                    val = 0
                    for ran in ranges:
                        if term[0] >= ran[0] and term[0] <= ran[1]:
                            oCol = ran[2]
                            val = self.domain[oCol][term[0]-ran[0]]
                            break
                    op = ">=" if term[1] == 1 else "<"
                    clause.add((oCol,val,op))
                newRow.append(clause)
            newRules.append([row[0],newRow])
        return newRules
    
    def getRules(self):
        rules = {}
        self.rulesstr = {}
        for row in self.rules:
            expr = "lambda x: ("
            clausules = []
            for clausule in row[1]:
                terms = []
                for term in clausule:
                    terms.append(f"x[{term[0]}] {term[2]} {term[1]}")
                temp = " or ".join(terms)
                clausules.append(f"({temp})")
            temp = expr + " and ".join(clausules) + ")"
            rules[row[0]] = eval(temp)
            self.rulesstr[row[0]] = temp
        return rules
    
    def predict(self,values):
        for key in self.rules2:
            if self.rules2[key](values):
                return key
        return -1
    
    def displayRule(self,labels):
        fun = f"def evaluate({' , '.join(labels)}):" + "\n"
        rules = []
        for row in self.rules:
            clausules = []
            for clausule in row[1]:
                terms = []
                for term in clausule:
                    terms.append(f"{labels[term[0]]} {term[2]} {term[1]}")
                temp = " or ".join(terms)
                clausules.append(f"({temp})")
            temp = " and ".join(clausules)
            rules.append("\t"+f"if {temp}:"+"\n\t\t"+f"return {row[0]}")
        temp = '\n'.join(rules)
        fun = f"{fun}{temp}"
        return fun