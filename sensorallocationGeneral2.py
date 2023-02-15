# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:22:47 2023

@author: hma2
"""

from mip import *
import numpy as np
import GridWorldV2

def LP(mdp, h, r_d, r_i):
    model = Model(solver_name=GRB)
    gamma = 0.95
    st_len = len(mdp.statespace)
    act_len_att = len(mdp.A)
    act_len_def = 2
    Z1 = -100
    Z2 = 100
    #Attacker and defender actions are binary
    pi1 = [[[model.add_var(var_type=BINARY) for i in range(act_len_def)] for j in range(act_len_att)] for k in range(st_len)] #Defender's policy 
    pi2 = [[model.add_var(var_type=BINARY) for i in range(act_len_att)] for j in range(st_len)] #Attacker's policy
    U1 = [model.add_var() for i in range(st_len)] #Defender's utility
    U2 = [model.add_var() for i in range(st_len)] #Attacker's Utility
    w1 = [[[model.add_var() for i in range(act_len_att)] for j in range(act_len_def)] for k in range(st_len)] #Defender's replacement
    w2 = [[[model.add_var() for i in range(act_len_att)] for j in range(act_len_def)] for k in range(st_len)] #Attacker's replacement
    U = mdp.U
    P = transfer_P(mdp)
    R_d = AssignReward(mdp.G, mdp.statespace, act_len_def, act_len_att, r_d)
    R_i = AssignReward(mdp.G, mdp.statespace, act_len_def, act_len_att, r_i)
    init = mdp.init
    model.objective = maximize(xsum(init[i] * U1[i] for i in range(st_len)))
    
    #If the state not in U, then defender takes action 0
    for i in range(st_len):
        if mdp.statespace[i] not in U:
            for j in range(act_len_att):
                model += pi1[i][j][0] == 1
                model += pi1[i][j][1] == 0
    #Only one action can be chosen in each state for both attacker and defender
    for i in range(st_len):
        model += xsum(pi2[i][j] for j in range(act_len_att)) == 1
    
    for i in range(st_len):
        for j in range(act_len_att):
            model += xsum(pi1[i][j][k] for k in range(act_len_def)) == 1
    #Number of sensors constraint
    model += xsum(pi1[i][j][1] for i in range(st_len) for j in range(act_len_att)) <= h
    
    #Utility Constraint
    for i in range(st_len):
        for act_att in range(act_len_att):
            model += U1[i] - xsum(pi1[i][act_att][act_def] * R_d[i][act_def][act_att] for act_def in range(act_len_def)) \
            - gamma * xsum(w1[i][act_def][act_att] for act_def in range(act_len_def)) - (1- pi2[i][act_att]) * Z2 <= 0
            
            model += U2[i] - xsum(pi1[i][act_att][act_def] * R_i[i][act_def][act_att] for act_def in range(act_len_def)) \
            - gamma * xsum(w2[i][act_def][act_att] for act_def in range(act_len_def)) - (1- pi2[i][act_att]) * Z2 <= 0
            
            model += U2[i] - xsum(pi1[i][act_att][act_def] * R_i[i][act_def][act_att] for act_def in range(act_len_def)) \
            - gamma * xsum(w2[i][act_def][act_att] for act_def in range(act_len_def)) >= 0
    
    #Replacement of w1 and w2
    for i in range(st_len):
        for act_def in range(act_len_def):
            for act_att in range(act_len_att):
                model += w1[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U1[j] for j in range(st_len)) >= Z1 * (1 - pi1[i][act_att][act_def])
                model += w1[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U1[j] for j in range(st_len)) <= Z2 * (1 - pi1[i][act_att][act_def])
                model += w1[i][act_def][act_att] >= Z1 * pi1[i][act_att][act_def]
                model += w1[i][act_def][act_att] <= Z2 * pi1[i][act_att][act_def] 
                
                model += w2[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U2[j] for j in range(st_len)) >= Z1 * (1 - pi1[i][act_att][act_def])
                model += w2[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U2[j] for j in range(st_len)) <= Z2 * (1 - pi1[i][act_att][act_def])
                model += w2[i][act_def][act_att] >= Z1 * pi1[i][act_att][act_def]
                model += w2[i][act_def][act_att] <= Z2 * pi1[i][act_att][act_def]
    
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        # sensorplace = [pi1[i][1].x for i in range(st_len)]
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value, sensorplace

def transfer_P(mdp):
    P = mdp.stotrans
    st_len = len(mdp.statespace)
    att_act_len = len(mdp.A)
    P_trans = np.zeros((st_len, 2, att_act_len, st_len))
    for i in range(st_len):
        st = mdp.statespace[i]
        for j in range(att_act_len):
            act_att = mdp.A[j]
            for st_, pro in mdp.stotrans[st][act_att].items():
                if st_ in mdp.statespace:
                    k = mdp.statespace.index(st_)
                    P_trans[i][0][j][k] = pro
    return P_trans

def AssignReward(G, statespace, actiondef, actionatt, reward):
    Reward = {}
    for i in range(len(statespace)):
        Reward[i] = {}
        state = statespace[i]
        for act_d in range(actiondef):
            Reward[i][act_d] = {}
            for act_a in range(actionatt):
                if state in G:
                    Reward[i][act_d][act_a] = reward
                else:
                    Reward[i][act_d][act_a] = 0
    return Reward

def main():
    goallist = [(1, 4)]
    gridworld = GridWorldV2.CreateGridWorld(goallist)
    gridworld.ChangeGoalTrans()
    h = 10 #Number of sensors allocation
    r_d = -1 #Defender's reward
    r_i = 1 #Attacker's reward
    DefenderValue, sensorplace = LP(gridworld, h, r_d, r_i)
    return DefenderValue, sensorplace

if __name__ == "__main__":
    DefenderValue, sensorplace = main()
    
    
                
                
                
    
                