import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams["animation.html"] = "jshtml"

def animate(data,cmap = "Set1",size=(20,10)):
    n_frames = len(data)
    dims = np.shape(data)
    colorsData = np.zeros(dims)
    for i in range(min(dims)):
        colorsData[i, i] = i
    fig = plt.figure(figsize=size)
    plot = plt.matshow(data[n_frames-1], fignum=0,cmap=cmap)

    def init():
        plot.set_data(data[0])
        return [plot]

    def update(j):
        plot.set_data(data[j])
        return [plot]

    anim = animation.FuncAnimation(fig, update, init_func = init, frames=n_frames, interval = 30, blit=True)

    return anim


def encodeKey(index,value):
    return "{},{}".format(index,value)

def decodeKey(key):
    return list(map(lambda x: int(x),key.split(",")))

#returns an expression to get the transformed coordinates 
# from the original dimensions to the 1 dimension flattened data
def getExpr(size):
    val = "lambda x:"
    lst = []
    if len(size) > 1:
        for i in range(1,len(size)):
            temp = "x[{}]".format(i-1)
            for j in range(i,len(size)):
                temp += "*{}".format(size[j])
            lst.append(temp)
    else:
        i = 0
    val += "+".join(lst)
    val += "+x[{}]".format(i)
    return eval(val)

dexpr = None

#returns an array with the position in the flattened data
#coords is an array with coordinate relative to the cell in the original dimensions
# size = np.shape(data)
def getNeighbors(cell,coords,size):
    global dexpr
    newCoords = []
    if dexpr == None:
        expr = getExpr(size)
        dexpr = expr
        #print(expr)
    else:
        expr = dexpr
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
            newCoord = expr(xi)
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
    #start = time.time()
    while (counters[0] >= 0):
        counters[count] -= 1
        #print("generator {}".format(time.time()-start))
        yield [int(i) for i in counters]
        #start = time.time()
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

#returns an array with the neighborhood
#expression = function to filter the neighborhood, receives a list of the indexes according to the dimension
#radious = array with the distance from each dimension                
def getNeighborhoodWLevels(radious,expression):
    neighborhood = {}
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
        validNeighbor = expression(stack[tIndex])
        if validNeighbor[0]:
            if str(validNeighbor[1]) not in neighborhood:
                neighborhood[str(validNeighbor[1])] = []
            neighborhood[str(validNeighbor[1])].append(stack[tIndex])
    for k in neighborhood:
        for i in range(dimensions-1,-1,-1):
            neighborhood[k].sort(key = lambda x: x[i])
    return neighborhood
    
def mooreWLevels(radious):
    expr = lambda x: (True, np.max(np.array(np.abs(x))))
    neighborhood = getNeighborhoodWLevels(radious,expr)
    return neighborhood

def vonNeumannWLevels(radious,distance):
    expr = lambda x: (manhattanDistance(x) <= distance, manhattanDistance(x))
    return getNeighborhoodWLevels(radious,expr)

def getLearningProblemWLevels(data,neighborhood,paddingType,paddingValue):
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
            values = {}
            for level in neighborhood:
                values[level] = []
                for neigh in neighborhood[level]:
                    if type(neigh) != list:
                        neighbors = getNeighbors(cell,[neigh],stateSize)
                    values[level].append({str(neigh): getNeighborsValue(currentState,neighbors,paddingType,paddingValue)[0] })
                    #values[level] = [values[level][key] for key in values[level]]
            problem[cls][str(values)] = values
    return problem

def information(problemWLevels,nlevels):
    info = {}
    for cls in problemWLevels:
        for level in range(nlevels):
            for key in problemWLevels[cls]:
                info[cls][level]
                
def getStr(levels,nlevels):
    res = {}
    middle = list(levels['0'][0].values())
    for i in range(1,nlevels+1): 
        i = str(i)
        left = levels[i][:(int)(len(levels[i])/2)]
        right = levels[i][(int)(len(levels[i])/2):]
        tmpleft = []
        tmpright = []
        for x in left:
            tmpleft += list(x.values())
        for x in right:
            tmpright += list(x.values())
        middle = tmpleft+copy(middle)+tmpright
        res[i] = middle
    return res

def strLevel(problem,nlevels):
    res = {}
    for cls in problem:
        res[cls] = {}
        for k in problem[cls]:
            temp = getStr(problem[cls][k],nlevels)
            for l in temp:
                if l not in res[cls]:
                    res[cls][l] = {}
                strtemp = str(temp[l])
                if strtemp not in res[cls][l]:
                    res[cls][l][strtemp] = ["",0]
                res[cls][l][strtemp][0] = temp[l]
                res[cls][l][strtemp][1] += 1 
    return res
    
def getTable(strlevels):
    tbl = {}
    for cls in strlevels:
        tbl[cls] = {}
        for lvl in strlevels[cls]:
            tbl[cls][lvl] = []
            for itm in strlevels[cls][lvl]:
                tbl[cls][lvl].append( strlevels[cls][lvl][itm])
            tbl[cls][lvl] = sorted(tbl[cls][lvl],key = lambda x:x[1],reverse = True)
    return tbl

def getTopFrequent(levels,top=10):
    mostFrequent = {}
    for cls in levels:
        mostFrequent[cls] = {}
        for lvl in levels[cls]:
            mostFrequent[cls][lvl] = levels[cls][lvl][:top]
    return mostFrequent

def getUniqueCount(levels):
    uniques = {}
    for cls in levels:
        uniques[cls] = {}
        for lvl in levels[cls]:
            uniques[cls][lvl] = 0
            for item in levels[cls][lvl]:
                if item[1] == 1:
                    uniques[cls][lvl] += 1
    return uniques

def getLevelClass(levels):
    res = {}
    for cls in levels:
        for lvl in levels[cls]:
            if lvl not in res:
                res[lvl] = {}
            for item in levels[cls][lvl]:
                itemstr = str(item[0])
                if itemstr not in res[lvl]:
                    res[lvl][itemstr] = {}
                if cls not in res[lvl][itemstr]:
                    res[lvl][itemstr][cls] = 0
                res[lvl][itemstr][cls] += item[1]
    return res


def getNeighborhoodAmbiguity(levels):
    ambiguity = {}
    for lvl in levels:
        ambiguity[lvl] = {}
        for item in levels[lvl]:
            if len(levels[lvl][item].keys()) > 1:
                ambiguity[lvl][item] = levels[lvl][item]
    return ambiguity

def getFrequencyByLocation(levels,nlstlevel):
    frequency = {}
    for cls in levels:
        frequency[cls] = {}
        for item in levels[cls][str(nlstlevel)]:
            for i,v in enumerate(item[0]):
                if i not in frequency[cls]:
                    frequency[cls][i] = {}
                if v not in frequency[cls][i]:
                    frequency[cls][i][v] = 0
                frequency[cls][i][v] += item[1]
    return frequency


def normalizeFrequency(frequency):
    normalized = {}
    for cls in frequency:
        normalized[cls] = {}
        for ind in frequency[cls]:
            normalized[cls][ind] = {}
            total = 0
            for v in frequency[cls][ind]:
                total += frequency[cls][ind][v]
            for v in frequency[cls][ind]:
                normalized[cls][ind][v] = frequency[cls][ind][v]/total
    return normalized

def getBDM(levels):
    bdm = BDM(ndim=1, nsymbols=4,raise_if_zero=False)
    res = {}
    for cls in levels:
        res[cls] = {}
        for lvl in levels[cls]:
            res[cls][lvl] = []
            for item in levels[cls][lvl]:
                tmp = np.array(item[0],dtype="int")
                #print(len(tmp))
                nbdm = 0
                try:
                    nbdm = bdm.nbdm(tmp)
                except:
                    nbdm = 0
                res[cls][lvl].append([item[0], nbdm])
            res[cls][lvl] = sorted(res[cls][lvl],key= lambda x: x[1],reverse = True)     
    return res

def getPlotFrequencies(frequencies):
    clsvals = {}
    ind = list(frequencies.keys())
    clss = list(frequencies.keys())
    for cls2 in clss:
        ind = list(frequencies[cls2].keys())
        clsvals[cls2] = {}
        for i in ind:
            for cls in clss:
                if cls not in clsvals[cls2]:
                    clsvals[cls2][cls] = []
                if cls not in frequencies[cls2][i]:
                    clsvals[cls2][cls].append(0)
                else:
                    clsvals[cls2][cls].append(frequencies[cls2][i][cls])
    for cls2 in clss:
        ind = list(frequencies[cls2].keys())
        bottoms = None
        bars = []
        subplt = plt.figure(figsize=(15,5))
        for cls in clsvals[cls2]:
            if bottoms is None:
                tmp = plt.bar(ind, clsvals[cls2][cls])
                bottoms = clsvals[cls2][cls]
            else:
                tmp = plt.bar(ind, clsvals[cls2][cls],bottom=bottoms)
                bottoms = np.sum([bottoms,clsvals[cls2][cls]],axis=0)
            bars.append(tmp)
        plt.title(cls2)
        plt.xticks(ind, ind)
        plt.legend(bars, [str(x) for x in list(clsvals[cls2].keys())])
        plt.show()