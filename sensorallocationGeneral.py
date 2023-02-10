# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:29:55 2023

@author: hma2
"""

from mip import *
import numpy as np
import GridWorldV2

def LP(mdp, h):
    model = Model(solver_name=GRB)
    gamma = 0.95
    st_len = len(mdp.statespace)
    act_len = len(mdp.A)
    y = [model.add_var() for i in range(st_len)]
    x = [model.add_var() for i in range(st_len * act_len)]   #State Action Frequency
    w = [model.add_var() for i in range(st_len * act_len)]   #w variable, used to replace x * y
    ww = [model.add_var() for i in range(st_len * st_len)]
    lmd = [model.add_var() for i in range(st_len * act_len)]  #KKT condition lambda, related to x
    mu = [model.add_var() for i in range(st_len)]   #KKT condition mu, related to flow constraint
    U = mdp.U
    R_d = DefenderReward(mdp.G, mdp.statespace, mdp.A)
    R_i = AttackerReward(mdp.G, mdp.statespace, mdp.A)
    init = mdp.init
    P = PTransition(mdp.stotrans, mdp.statespace, mdp.A, st_len, act_len)
    Z = 1   #Constant used in McCormick in equality of w
    Z2 = 1 #Constant used in McCormick in equality of ww
    #maximize the defender's reward
    model.objective = maximize(xsum(R_d[i] * x[i] for i in range(st_len * act_len)))
    for i in range(st_len):
        if mdp.statespace[i] not in U:
            model += y[i] == 0
    
    #Number of sensors constraint
    model += xsum(y[j] for j in range(st_len)) <= h
    
    #constraint for w, McCormick inequality
    for i in range(st_len):
        for j in range(act_len):
            model += w[i * act_len + j] - x[i * act_len + j] + Z * (1 - y[i]) >= 0
            model += w[i * act_len + j] - x[i * act_len + j] - Z * (1 - y[i]) <= 0
            model += w[i * act_len + j] + Z * y[i] >= 0
            model += w[i * act_len + j] - Z * y[i] <= 0
            
    #constraint for ww, McCormick inequlity, test part
    for i in range(st_len):
        for j in range(st_len):
            model += ww[i * st_len + j] - mu[j] + Z2 * (1 - y[i]) >= 0
            model += ww[i * st_len + j] - mu[j] - Z2 * (1 - y[i]) <= 0
            model += ww[i * st_len + j] + Z2 * y[i] >= 0
            model += ww[i * st_len + j] - Z2 * y[i] <= 0
    #Test ends here
            
    #SOS1 constraint
    for i in range(st_len * act_len):
        model.add_sos([(x[i], 1),(lmd[i], 1)], 1)
        model += lmd[i] >= 0
        model += x[i] >= 0
    
    #Primal Feasibility
    for i in range(st_len):
        model += xsum(x[i * act_len + j] for j in range(act_len)) - gamma * \
            xsum(P[j // act_len][j % act_len][i] * x[j] for j in range(st_len * act_len)) \
            + gamma * xsum(P[j // act_len][j % act_len][i] * w[j] for j in range(st_len * act_len)) \
            - init[i] == 0

    #Lagrangian function derivation
    for i in range(st_len * act_len):
        st_index = i // act_len
        act_index = i % act_len
        # model += -R_i[i] + lmd[i] + mu[st_index] - gamma * (1 - y[st_index]) * xsum(mu[j] * P[st_index][act_index][j] for j in range(st_len)) == 0
        
        model += -R_i[i] + lmd[i] + mu[st_index] - gamma * xsum(mu[j] * P[st_index][act_index][j] for j in range(st_len)) + \
            gamma * xsum(ww[st_index * st_len + j] * P[st_index][act_index][j] for j in range(st_len))== 0   #Use ww
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        sensorplace = [y[i].x for i in range(st_len)]
        occup = [x[i].x for i in range(st_len * act_len)]
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value, sensorplace, occup

def PTransition(trans, statespace, actionspace, st_len, act_len):
    P_Matrix = np.zeros((st_len, act_len, st_len))
    for st in trans.keys():
        st_index = statespace.index(st)
        for act in trans[st].keys():
            act_index = actionspace.index(act)
            for st_, value in trans[st][act].items():
                if st_ in statespace:    #Not Sink
                    P_Matrix[st_index][act_index][statespace.index(st_)] = value
    return P_Matrix

def DefenderReward(G, statespace, actionspace):
    r = -1
    R_d = np.zeros(len(statespace) * len(actionspace))
    for g in G:
        g_index = statespace.index(g)
        for act in range(len(actionspace)):
            R_d[g_index * len(actionspace) + act] = r
    return R_d

def AttackerReward(G, statespace, actionspace):
    r = 1
    R_i = np.zeros(len(statespace) * len(actionspace))
    for g in G:
        g_index = statespace.index(g)
        for act in range(len(actionspace)):
            R_i[g_index * len(actionspace) + act] = r
    return R_i
    

if __name__ == "__main__":
    goallist = [(1, 4)]
    gridworld = GridWorldV2.CreateGridWorld(goallist)
    gridworld.ChangeGoalTrans()
    def_value, sensor_place, occup = LP(gridworld, 2)
    
    