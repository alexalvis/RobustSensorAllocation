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
    lmd = [model.add_var() for i in range(st_len * act_len)]  #KKT condition lambda, related to x
    mu = [model.add_var() for i in range(st_len)]   #KKT condition mu, related to flow constraint
    U = mdp.U
    R_d = np.zeros(st_len * act_len)
    R_i = np.zeros(st_len * act_len)
    init = mdp.init
    Z = 2   #Constant used in McCormick in equality
    #maximize the defender's reward
    model.objective = maximize(xsum(R_d[i] * x[i] for i in range(st_len * act_len)))
    for i in range(stlen):
        if mdp.statespace[i] not in U:
            model += y[i] == 0
    
    #Number of sensors constraint
    model += xsum(y[j] for j in range(stlen)) <= h
    
    #constraint for w, McCormick inequality
    for i in range(st_len):
        for j in range(act_len):
            model += w[i * act_len + j] - x[i * act_len + j] + Z * (1 - y[i]) >= 0
            model += w[i * act_len + j] - x[i * act_len + j] - Z * (1 - y[i]) <= 0
            model += w[i * act_len + j] + Z * y[i] >= 0
            model += w[i * act_len + j] - Z * y[i] <= 0
    #SOS1 constraint
    for i in range(st_len * act_len):
        model.add_sos([(x[i], 1),(lmd[i], 1)], 1)
        model += lmd[i] >= 0
        model += x[i] >= 0
    
    for i in range(st_len):
        model += xum(x[i * act_len + j] for j in range(act_len)) - gamma * xsum(P[j // act_len][j % act_len][i] * x[j] for j in range(st_len)) + \
        gamma * xsum(P[j // act_len][j % act_len][i] * w[j] for j in range(st_len)) - init[i] == 0
    
    for i in range(st_len * act_len):
        model += -R_i[i] + lmd[i] + (something) == 0
    
if __name__ == "__main__":
    
    