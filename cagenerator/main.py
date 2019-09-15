import cellpylib as cpl
import numpy as np
import os
# initialize a 60x60 2D cellular automaton 
cellular_automaton = cpl.init_simple2d(60, 60)

# evolve the cellular automaton for 30 time steps, 
#  applying totalistic rule 126 to each cell with a Moore neighbourhood
cellular_automaton = cpl.evolve2d(cellular_automaton, timesteps=30, neighbourhood='Moore',
                                  apply_rule=lambda n, c, t: cpl.totalistic_rule(n, k=2, rule=126))
dirName = "data"
if not os.path.exists(dirName):
	os.mkdir(dirName)
for i in range(30):
	x = cellular_automaton[i]
	np.savetxt("{}/step{}.csv".format(dirName,i),x,fmt='%d', delimiter=",")

cpl.plot2d_animate(cellular_automaton)