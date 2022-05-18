"""
code written by Harshit Batra 
"""

import random
from turtle import position
import numpy
import math
from solution import solution
import time

"""
pseudo code:
Start
    Initializing
    Calculate fitness function of all solutions
    While the convergence criterion is not satisfied
        Obtain Xave which is the mean value of all the solutions
        Set the solution with the worst fitness as Xw
        If fitness of Xave is better than the fitness of Xw
            Xw = Xave
        End
        For i=1: Npop (each solution)
            Select Xj randomly which is not equal to Xi
            Compare the solutions Xi, Xj, and Xave and set the best solution as the Xbest, the second best
            Solution as Xmedium, and third solution as Xworst
            Update Xi based on equations 1-4
            Check the constraints and substitute the new solution with the old one based on Eq. 5
        End
        For i=1:Npop (each solution)
            For j=1:Nv (number of variables)
                Update the solutions based on 6
                Check the constraints and substitute the new solution with the old one based on Eq. 7
            End
        End
    End
Stop
"""


def MROM(objf, lb, ub, dim, SearchAgents_no, Max_iter, m=7 ):

    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5

    # initialize xw, xavg, and xmed
    Xw_pos = numpy.zeros(dim)
    Xw_score = float("inf")

    Xavg_pos = numpy.zeros(dim)
    Xavg_score = float("inf")

    best_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('MROM is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, Max_iter):
        index = 0 # index of the worst individual
        for i in range(0, SearchAgents_no):

            Xavg_pos = Positions.mean(0)

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])
            Xavg_score = objf(Xavg_pos)
            # Update Xw
            if fitness > Xw_score:
                Xw_score = fitness
                Xw_pos = Positions[i, :].copy()
                index = i
            if fitness < best_score:
                best_score = fitness # update best score    
        
        # Update Xw basewd on xavg
        if Xavg_score < Xw_score:
            Xw_score = Xavg_score
            Xw_pos = Xavg_pos.copy()
            Positions[index, :] = Xw_pos.copy()


        # Update the Position of search agents 
        #phase 1 
        for i in range(0, SearchAgents_no):
            j = numpy.random.randint(0, SearchAgents_no-1) # j!=1
            if j>= i :
                j = j+1
            fitness_i = objf(Positions[i, :])
            fitness_j = objf(Positions[j, :])
            if fitness_i > fitness_j:
                if fitness_i > Xavg_score:
                    x_worst = Positions[i, :].copy()
                    if fitness_j > Xavg_score:
                        x_best = Xavg_pos.copy()
                        x_medium = Positions[j, :].copy()
                    else:
                        x_best = Positions[j, :].copy()
                        x_medium = Xavg_pos.copy()
                else :
                    x_worst = Xavg_pos.copy()
                    x_best = Positions[j, :].copy()
                    x_medium = Positions[i, :].copy()
            else:
                if fitness_j > Xavg_score:
                    x_worst = Positions[j, :].copy()
                    if fitness_i > Xavg_score:
                        x_best = Xavg_pos.copy()
                        x_medium = Positions[i, :].copy()
                    else:
                        x_best = Positions[i, :].copy()
                        x_medium = Xavg_pos.copy()
                else :
                    x_worst = Xavg_pos.copy()
                    x_best = Positions[i, :].copy()
                    x_medium = Positions[j, :].copy()
            # Update the Position of search agents
            #phase 1 
            Xt = numpy.zeros(dim)

            for k in range(dim):
                Xt[k] = x_medium[k]-x_worst[k] #eq 1 in the paper
            
            T = i/Max_iter

            metal_ratio = ( m + abs(math.sqrt(m**2 + 4)) ) / 2 # goldern ratio 

            ft = metal_ratio*(metal_ratio**T - (m-metal_ratio)**T) / abs(math.sqrt(m**2 + 4)) # eq 2 in the paper

            Xnew = (1-ft)*x_best + numpy.random.rand()*ft*Xt # eq 3 in the paper
            for j in range(dim):
                Xnew[j] = numpy.clip(Xnew[j], lb[j], ub[j])

            if objf(Xnew) < objf(Positions[i, :]):
                Positions[i, :] = Xnew.copy()   # eq 4 in the paper
            
        #phase 2
        for i in range(0, SearchAgents_no):

            j = numpy.random.randint(0, SearchAgents_no-1) # j!=1
            if j>= i :
                j = j+1

            fitness_i = objf(Positions[i, :])
            fitness_j = objf(Positions[j, :])

            if fitness_i > fitness_j:
                if fitness_i > Xavg_score:
                    x_worst = Positions[i, :].copy()
                    if fitness_j > Xavg_score:
                        x_best = Xavg_pos.copy()
                        x_medium = Positions[j, :].copy()
                    else:
                        x_best = Positions[j, :].copy()
                        x_medium = Xavg_pos.copy()
                else :
                    x_worst = Xavg_pos.copy()
                    x_best = Positions[j, :].copy()
                    x_medium = Positions[i, :].copy()
            else:
                if fitness_j > Xavg_score:
                    x_worst = Positions[j, :].copy()
                    if fitness_i > Xavg_score:
                        x_best = Xavg_pos.copy()
                        x_medium = Positions[i, :].copy()
                    else:
                        x_best = Positions[i, :].copy()
                        x_medium = Xavg_pos.copy()
                else :
                    x_worst = Xavg_pos.copy()
                    x_best = Positions[i, :].copy()
                    x_medium = Positions[j, :].copy()

            # Update the Position of search agents
            Xnew = Positions[i,:] + (1/metal_ratio)*numpy.random.rand()*(x_best-x_worst) # eq 5 in the paper
            for j in range(dim):
                Xnew[j] = numpy.clip(Xnew[j], lb[j], ub[j])

            if objf(Xnew) < objf(Positions[i, :]):
                Positions[i, :] = Xnew.copy() 
                
        Convergence_curve[l] = best_score

        if l % 1 == 0:
            print(
            ["At iteration " + str(l) + " the best fitness is " + str(best_score)]
            )
    print(len(Convergence_curve))
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "MROM"
    s.objfname = objf.__name__

    return s