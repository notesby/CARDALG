import numpy as np
import random
from copy import copy
from joblib import Parallel, delayed,parallel_backend
import helpers
import matplotlib.pyplot as plt

class CAModel():
    def __init__(self,rules,neighborhood,paddingType,paddingValue):
        self.paddingType = paddingType
        self.paddingValue = paddingValue
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

class Testing1():
    def __init__(self,data,model,verbose=0):
        self.data = data
        self.model = model
        self.verbose = verbose
    
    def testRow(self,row_prev,row,row_pred):
        totalMatchDynamic = 0
        totalMatchStatic = 0
        totalStatic = 0
        totalDynamic = 0
        accuracy = 0
        for cell0,cell1,cell2 in zip(row_prev,row,row_pred):
            totalMatchDynamic += (cell0 != cell1 and cell1 == cell2)
            totalMatchStatic += (cell0 == cell1 and cell1 == cell2)
            totalStatic += (cell0 == cell1)
            totalDynamic += (cell0 != cell1)
            accuracy += (cell1 == cell2)
        totalStatic = totalStatic if totalStatic > 0 else 0
        totalDynamic = totalDynamic if totalDynamic > 0 else 0
        return [totalStatic,totalDynamic,totalMatchStatic,totalMatchDynamic,accuracy]
            
    def run(self):
        res = []
        backend = "threading"
        with Parallel(n_jobs=4,backend=backend) as parallel:
            if self.verbose >= 1: print("Testing:")
            for i in range(1,len(self.data)):
                if self.verbose >= 1: print(f"state {i}/{len(self.data)}")
                prediction = next(self.model.run(self.data[i-1],1))
                rowsTotal = parallel(delayed(self.testRow)(row0,row1,row2) for row0,row1,row2 in zip(self.data[i-1],self.data[i],prediction[1]))
                summation = np.sum(rowsTotal,axis=0)
                if self.verbose >= 2: 
                    print(f"Static : {summation[2]}/{summation[0]}")
                    print(f"Dynamic : {summation[3]}/{summation[1]}")
                    print(f"Accuracy: {summation[4]} / {self.data[i].size}")
                totalStatic = summation[2]/summation[0] if summation[0] > 0 else 0
                totalDynamic = summation[3]/summation[1] if summation[1] > 0 else 0
                accuracy = summation[4] / self.data[i].size
                res.append([totalStatic,totalDynamic,accuracy])
        self.result = np.array(res)
    
    def plot(self,figsize=(15,10)):
        plt.figure(figsize=figsize)
        plt.plot(range(len(self.result)),self.result[:,0],label="Trivial (static behaviour)")
        plt.plot(range(len(self.result)),self.result[:,1],label="Not trivial (Not static behaviour)")
        plt.plot(range(len(self.result)),self.result[:,2],label="Accuracy (General behaviour)")
        plt.yticks(np.arange(0, 1, step=0.05))
        plt.ylabel("Accuracy")
        plt.xlabel("Simulation step")
        plt.grid(True)
        plt.legend()
        return plt