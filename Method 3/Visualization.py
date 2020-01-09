import numpy as np
import json
import os
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

class Visualization():
    def __init__(self,path):
        self.path = path
        self.data = []
        self.nodes = {}
        self.loadData()
        self.getNodesByPopulation()
        
    def loadData(self):
        for file in os.listdir(self.path):
            if ".json" in file and file != "config.json":
                path = "{}/{}".format(self.path,file)
                with open(path) as json_file:
                    self.data.append(json.load(json_file))
        self.data = sorted(self.data,key = lambda x : x["generation"])
    
    def getFitnessByPopulation(self,measureType):
        fitness = {}
        for world in self.data:
            for population in world["populations"]:
                goal = population["goal"]
                if goal not in fitness:
                    fitness[goal] = []
                fitness[goal].append([population["generation"],population[measureType]])
        for goal in fitness:
            fitness[goal] = np.array(sorted(fitness[goal],key=lambda x:x[0]))
        return fitness
    
    def getFitnessGraph(self,popid):
        avg = self.getFitnessByPopulation("average")
        best = self.getFitnessByPopulation("best")
        median = self.getFitnessByPopulation("median")
        fig, ax = plt.subplots(figsize=(20,10))
        ax.plot(avg[popid][:,0],avg[popid][:,1],"b")
        ax.plot(best[popid][:,0],best[popid][:,1],"r")
        ax.plot(median[popid][:,0],median[popid][:,1],"g")
        ax.set_title('Fitness of class {}'.format(popid),fontsize=18,color='orange')
        ax.set_xlabel('Generations',fontsize=18,color='orange')
        ax.set_ylabel('Fitness',fontsize=18,color='orange')
        plt.legend(('Mean', 'Best', 'Median'),loc='upper right')
        ax.tick_params(labelsize=18,labelcolor='orange')
        return ax
    
    def getNodesByPopulation(self):
        nodes = {}
        for world in self.data:
            for population in world["populations"]:
                goal = population["goal"]
                if goal not in nodes:
                    nodes[goal] = {}
                for chromosome in population["chromosomes"]:
                    nodes[goal][chromosome["uuid"]] = chromosome
        self.nodes = nodes
        
    def getDendogram(self,idpop):
        labels = list(self.nodes[idpop].keys())
        idkey = {}
        for i,key in enumerate(self.nodes[idpop].keys()):
            idkey[key] = i
        matrix = np.zeros(shape=(len(labels),len(labels)))
        for node in self.nodes[idpop]:
            index = idkey[self.nodes[idpop][node]["uuid"]]
            parents = [idkey[x] for x in self.nodes[idpop][node]["parents"]]
            if len(parents) > 0:
                for parent in parents:
                    matrix[index][parent] = 1
        fig = ff.create_dendrogram(matrix, labels=labels)
        fig.update_layout(width=2500, height=1024)
        return fig
        
    def getAncestryGraph(self,idpop):
        idkey = {}
        labels = list(self.nodes[idpop].keys())
        source = []
        target = []
        value = []
        for i,key in enumerate(self.nodes[idpop].keys()):
            idkey[key] = i
        
        labels = [i for i in range(len(labels))]
        labels.append("no parent")
        for node in self.nodes[idpop]:
            parents = [idkey[x] for x in self.nodes[idpop][node]["parents"]]
            if len(parents) > 0:
                source += parents
                target += [idkey[self.nodes[idpop][node]["uuid"]] for i in range(len(parents))]
                value += [self.nodes[idpop][node]["generation"] for i in range(len(parents))]
            else:
                source.append(len(labels)-1)
                target.append(idkey[self.nodes[idpop][node]["uuid"]])
                value.append(self.nodes[idpop][node]["generation"])
            
        fig = go.Figure(data=[go.Sankey(
                      orientation = "h",
        node = dict(
                      pad = 15,
                      thickness = 20,
                      line = dict(color = "black", width = 0.5),
                      label = labels,
                      color = "blue"
                    ),
                    link = dict(
                      source = source, 
                      target = target,
                      value = value
                  ))],layout=dict(width = 2500,
                      height = 1024))

        fig.update_layout(title_text="Ancestry", font_size=10)
        return fig