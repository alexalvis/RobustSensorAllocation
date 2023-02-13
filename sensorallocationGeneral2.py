# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:22:47 2023

@author: hma2
"""

from mip import *
import numpy as np
import GridWorldV2

def LP(mdp, h):
    model = Model(solver_name=GRB)
    gamma = 0.95
    st_len = len(mdp.statespace)
    act_len_att = len(mdp.A)
    act_len_def = 2
    Z = 1
    #Attacker and defender actions are binary
    pi1 = [[model.add_var(var_type=BINARY) for i in range(act_len_def)] for j in range(st_len)] #Defender's policy 
    pi2 = [[model.add_var(var_type=BINARY) for i in range(act_len_att)] for j in range(st_len)] #Attacker's policy
    U1 = [model.add_var() for i in range(st_len)] #Defender's utility
    U2 = [model.add_var() for i in range(st_len)] #Attacker's Utility
    w1 = [[[model.add_var() for i in range(st_len)] for j in range(act_len_def)] for k in range(st_len)] #Defender's replacement
    w2 = [[[model.add_var() for i in range(st_len)] for j in range(act_len_def)] for k in range(st_len)] #Attacker's replacement
    U = mdp.U
    R_d = DefenderReward(mdp.G, mdp.statespace, mdp.A)
    R_i = AttackerReward(mdp.G, mdp.statespace, mdp.A)
    init = mdp.init
    model.objective = maximize(xsum(R_d[i] * U1[i] for i in range(st_len * act_len)))
    
    #If the state not in U, then defender takes action 0
    for i in range(st_len):
        if mdp.statespace[i] not in U:
            model += pi1[i][0] == 1
    #Only one action can be chosen in each state for both attacker and defender
    for i in range(st_len):
        model += xsum(pi1[i][j] for j in range(act_len_def)) == 1
        model += xsum(pi2[i][j] for j in range(act_len_att)) == 1
    #Number of sensors constraint
    model += xsum(pi1[i][1] for i in range(st_len)) <= h
    
    #Utility Constraint
    for i in range(st_len):
        for act_att in range(act_len_att):
            model += U1 - xsum(pi1[i][act_def] * R_d[i][act_def][act_att] for act_def in range(act_len_def)) \
            - gamma * xsum(w1[i][act_def][act_att] for act_def in range(act_len_def)) - (1- pi2[i][act_att]) * Z <= 0
            
            model += U2 - xsum(pi1[i][act_def] * R_i[i][act_def][act_att] for act_def in range(act_len_def)) \
            - gamma * xsum(w2[i][act_def][act_att] for act_def in range(act_len_def)) - (1- pi2[i][act_att]) * Z <= 0
            
            model += U2 - xsum(pi1[i][act_def] * R_i[i][act_def][act_att] for act_def in range(act_len_def)) \
            - gamma * xsum(w2[i][act_def][act_att] for act_def in range(act_len_def)) >= 0
    
    #Replacement of w1 and w2
    for i in range(st_len):
        for act_def in range(act_len_def):
            for act_att in range(act_len_att):
                model += w1[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U1[j] for j in range(st_len)) + Z * (1-pi1[i][act_def]) >= 0
                model += w1[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U1[j] for j in range(st_len)) - Z * (1-pi1[i][act_def]) <= 0
                model += w1[i][act_def][act_att] + Z * pi1[i][act_def] >= 0
                model += w1[i][act_def][act_att] - Z * pi1[i][act_def] <= 0
                
                model += w2[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U2[j] for j in range(st_len)) + Z * (1-pi1[i][act_def]) >= 0
                model += w2[i][act_def][act_att] - xsum(P[i][act_def][act_att][j] * U2[j] for j in range(st_len)) - Z * (1-pi1[i][act_def]) <= 0
                model += w2[i][act_def][act_att] + Z * pi1[i][act_def] >= 0
                model += w2[i][act_def][act_att] - Z * pi1[i][act_def] <= 0
    
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        sensorplace = [pi1[i][1].x for i in range(st_len)]
                