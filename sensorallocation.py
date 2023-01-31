# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:52:29 2023

@author: hma2
"""

from mip import *
import numpy as np
import GridWorldV2  

def LP(k, h, m, M, mdplist, v_i):
    #k number of attackers, h number of sensors constraint
    #m lower bound of big M method, M upper bound of big M method
    model = Model(solver_name=GRB)
    gamma = 0.95
    mdp = mdplist[0]
    stlen = len(mdp.statespace)
    init = np.zeros(stlen)
    y = model.add_var() #y is regret
    V = [[model.add_var(lb = 0, ub = 1) for j in range(stlen)] for i in range(k)]
    #V is the value under x configuration
    x = [m.add_var(var_type=BINARY) for i in range(st_len)]
    #x is the sensor placement configuration
    w = [[model.add_var(lb = 0, ub = 1) for j in range(len(mdp.st2st))] for i in range(k)]
    U = mdp.U  #Read from mdp directly
    #v_i is the optimal value under different attackers, given as input
    model.objective = minimize(y)
    for i in range(k):
        #minimize regret
        model += y - (xsum(init[j] * V[i][j] for j in range(stlen)) - v_i[i]) >= 0
    #State outside U can not be sensors
    for j in range(stlen):
        if mdp.statespace[j] not in U:
            model += x[j] == 0
    #number of ids is limited
    model += xsum(x[j] for j in range(stlen)) <= h
    for i in range(k):
        mdp = mdplist[i]
        #get the corresponding mdp
        for j in range(stlen):
            #current state: j
            #next state: ns
            if mdp.statespace[j] in G:
                model += V[i][j] == 1
        
            for a in mdp.A:
                model += V[i][j] >= gamma * xsum(P[j][a][ns] * w[mdp.st2st.index((j, ns))] for ns in mdp.stotrans[j][a].keys())
            
            for ns in mdp.stotrans[j][a].keys():
                #index of (s, s'): k
                k = mdp.st2st.index((j, ns))
                model += w[i][k] >= m * (1 - x[j])
                model += w[i][k] <= M * (1 - x[j])
                model += w[i][k] - V[i][mdp.statespace.index(ns)] >= m * x[j]
                model += w[i][k] - V[i][mdp.statespace.index(ns)] <= M * x[j]
                
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        x_res = [x[i].x for i in range(st_len)]
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value, x_res

def sub_solver(h, m, M, mdp):
    #h number of sensors constraint, m lower bound of big M, M upper bound of big M
    #mdp is the current mdp
    model = Model(solver_name=GRB)
    gamma = 0.95
    
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value

def main(k, h, m, M):
    mdplist = ...
    v_i = []
    for mdp in mdplist():
        v_opt = sub_solver(h, m, M, mdp)
        v_i.append(v_opt)
    regret, ids_config = LP(k, h, m, M, mdplist, v_i)
    return regret, ids_config

if __name__ == "__main__":
    k = 2  #Number of attacker types
    h = 2  #Number of sensor constraints
    m = 0  #Lower bound of big M method
    M = 1  #Upper bound of big M method
    regret, ids_config = main(k, h, m, M)
    
    
    