# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:52:29 2023

@author: hma2
"""

from mip import *
import numpy as np
import GridWorldV2  

def LP(num_att, h, m, M, mdplist, v_i, r_i):
    #k number of attackers, h number of sensors constraint
    #m lower bound of big M method, M upper bound of big M method
    model = Model(solver_name=GRB)
    # model.Params.PoolSearchMode = 2
    # model.setParam(GRB.Param.PoolSearchMode, 2)
    gamma = 0.95
    mdp = mdplist[0]
    stlen = len(mdp.statespace)
    init = mdplist[0].init
    y = model.add_var() #y is regret
    V = [[model.add_var(lb = 0) for j in range(stlen)] for i in range(num_att)]
    #V is the value under x configuration
    x = [model.add_var(var_type=BINARY) for i in range(stlen)]
    #x is the sensor placement configuration
    w = [[model.add_var(lb = 0) for j in range(len(mdplist[i].stast))] for i in range(num_att)]
    U = mdp.U  #Read from mdp directly
    #v_i is the optimal value under different attackers, given as input
    model.objective = minimize(y)
    for i in range(num_att):
        #minimize regret
        model += y - (xsum(init[j] * V[i][j] for j in range(stlen)) - v_i[i]) >= 0
    #State outside U can not be sensors
    for j in range(stlen):
        if mdp.statespace[j] not in U:
            model += x[j] == 0
    #number of ids is limited
    model += xsum(x[j] for j in range(stlen)) <= h
    for i in range(num_att):
        mdp = mdplist[i]
        #get the corresponding mdp
        for j in range(stlen):
            #current state: j
            #next state: ns
            if mdp.statespace[j] in mdp.G:
                model += V[i][j] == r_i[i]
        
            for a in mdp.A:
                model += V[i][j] >=  xsum(mdp.stotrans[mdp.statespace[j]][a][ns] * w[i][mdp.stast.index((mdp.statespace[j], a, ns))] for ns in mdp.stotrans[mdp.statespace[j]][a].keys())
            
                for ns in mdp.stotrans[mdp.statespace[j]][a].keys():
                    #index of (s, s'): k
                    k = mdp.stast.index((mdp.statespace[j], a, ns))
                    model += w[i][k] >= m * (1 - x[j])
                    model += w[i][k] <= M * (1 - x[j])
                    model += w[i][k] - gamma * V[i][mdp.statespace.index(ns)] >= m * x[j]
                    model += w[i][k] - gamma * V[i][mdp.statespace.index(ns)] <= M * x[j]
                
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        x_res = [x[i].x for i in range(stlen)]
        for i in range(num_att):
            print(xsum(init[j] * V[i][j].x for j in range(stlen)) - v_i[i])
        for j in range(stlen):
            print(V[1][j].x)
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value, x_res

def sub_solver(h, m, M, mdp, reward):
    #h number of sensors constraint, m lower bound of big M, M upper bound of big M
    #mdp is the current mdp
    model = Model(solver_name=GRB)
    stlen = len(mdp.statespace)
    init = mdp.init
    gamma = 0.95
    v = [model.add_var() for i in range(stlen)]  # V(s)
    x = [model.add_var(var_type=BINARY) for i in range(stlen)]  #  x(s)
    w = [model.add_var() for j in range(len(mdp.stast))]
    U = mdp.U
    model.objective = minimize(xsum(init[i] * v[i] for i in range(stlen)))
    
    for j in range(stlen):
        if mdp.statespace[j] not in U:
            model += x[j] == 0
    #number of ids is limited
    model += xsum(x[j] for j in range(stlen)) <= h
    
    for j in range(stlen):
        #current state index: j
        #next state: ns
        if mdp.statespace[j] in mdp.G:
            model += v[j] == reward
        
        for a in mdp.A:
            model += v[j] >=  xsum(mdp.stotrans[mdp.statespace[j]][a][ns] * w[mdp.stast.index((mdp.statespace[j], a, ns))] 
                                          for ns in mdp.stotrans[mdp.statespace[j]][a].keys())
        
            for ns in mdp.stotrans[mdp.statespace[j]][a].keys():
                #index of (s, s'): k
                k = mdp.stast.index((mdp.statespace[j], a, ns))
                model += w[k] >= m * (1 - x[j])
                model += w[k] <= M * (1 - x[j])
                model += w[k] - gamma * v[mdp.statespace.index(ns)] >= m * x[j]
                model += w[k] - gamma * v[mdp.statespace.index(ns)] <= M * x[j]
    
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        x_res = [x[i].x for i in range(stlen)]
        v_res = [v[i].x for i in range(stlen)]
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value, x_res, v_res

def evaluate_sensor(mdp, sensorplace, reward):
    model = Model(solver_name=GRB)
    gamma = 0.95
    stlen = len(mdp.statespace)
    init = mdp.init
    V = [model.add_var(lb = 0) for j in range(stlen)]
    model.objective = minimize(xsum(init[i] * V[i] for i in range(stlen)))
    for i in range(stlen):
        if mdp.statespace[i] in mdp.G:
            model += V[i] == reward
        elif sensorplace[i] == 1:
            model += V[i] == 0
        else:
            for a in mdp.A:
                model += V[i] >= xsum(gamma * mdp.stotrans[mdp.statespace[i]][a][ns] * \
                                      V[mdp.statespace.index(ns)] for ns in mdp.stotrans[mdp.statespace[i]][a].keys())
    status = model.optimize()  # Set the maximal calculation time
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        V_res = [V[i].x for i in range(stlen)]
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value, V_res
    
    
def main(k, h, m, M):
    mdplist = ...
    v_i = []
    for mdp in mdplist():
        v_opt = sub_solver(h, m, M, mdp)
        v_i.append(v_opt)
    regret, ids_config = LP(k, h, m, M, mdplist, v_i)
    return regret, ids_config

if __name__ == "__main__":
    num_att = 2  #Number of attacker types
    h = 2  #Number of sensor constraints
    m = -1  #Lower bound of big M method
    M = 3  #Upper bound of big M method
    # regret, ids_config = main(k, h, m, M)
    goallist1 = [(1, 4)]
    goallist2 = [(4, 4)]
    mdp1 = GridWorldV2.CreateGridWorld(goallist1)
    mdp2 = GridWorldV2.CreateGridWorld(goallist2)
    v1, x1, v_spec_1 = sub_solver(h, m, M, mdp1, 3)
    v2, x2, v_spec_2 = sub_solver(h, m, M, mdp2, 3)
    mdplist = [mdp1, mdp2]
    vlist = [v1, v2]
    reward = [3, 3]
    regret, x_regret = LP(num_att, h, m, M, mdplist, vlist, reward)
    x1_count = np.zeros(23)
    x1_count[2] = 1
    x1_count[6] = 1
    objectValue, V_spec = evaluate_sensor(mdp1, x1_count, 3)
    
    
    