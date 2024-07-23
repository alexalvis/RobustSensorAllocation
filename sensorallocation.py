# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:52:29 2023

@author: hma2
"""

from mip import *
import numpy as np
import GridWorldV2
import time

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
                model += V[i][j] == r_i[i][mdp.G.index(mdp.statespace[j])]
        
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
            print(xsum(init[j] * V[i][j].x for j in range(stlen)))
        # for j in range(stlen):
            # print(V[1][j].x)
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
            model += v[j] == reward[mdp.G.index(mdp.statespace[j])]
        
        for a in mdp.A:
            model += v[j] >=  xsum(gamma * mdp.stotrans[mdp.statespace[j]][a][ns] * w[mdp.stast.index((mdp.statespace[j], a, ns))] 
                                          for ns in mdp.stotrans[mdp.statespace[j]][a].keys())
        
            for ns in mdp.stotrans[mdp.statespace[j]][a].keys():
                #index of (s, s'): k
                k = mdp.stast.index((mdp.statespace[j], a, ns))
                model += w[k] >= m * (1 - x[j])
                model += w[k] <= M * (1 - x[j])
                model += w[k] -  v[mdp.statespace.index(ns)] >= m * x[j]
                model += w[k] -  v[mdp.statespace.index(ns)] <= M * x[j]
    
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
            model += V[i] == reward[mdp.G.index(mdp.statespace[i])]
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
    
    
def main():
    num_att = 2  #Number of attacker types
    h = 2  #Number of sensor constraints
    m = -10  #Lower bound of big M method
    M = 10  #Upper bound of big M method
    # regret, ids_config = main(k, h, m, M)
    goallist1 = [(1, 4)]
    goallist2 = [(4, 4)]
    mdp1 = GridWorldV2.CreateGridWorld(goallist1)
    mdp2 = GridWorldV2.CreateGridWorld(goallist2)
    v1, x1, v_spec_1 = sub_solver(h, m, M, mdp1, [1])
    v2, x2, v_spec_2 = sub_solver(h, m, M, mdp2, [0.95])
    mdplist = [mdp1, mdp2]
    vlist = [v1, v2]
    reward = [[1], [0.95]]
    regret, x_regret = LP(num_att, h, m, M, mdplist, vlist, reward)
    print(x_regret)
#    x1_count = np.zeros(23)
#    x1_count[7] = 1
#    x1_count[15] = 1
#    objectValue, V_spec = evaluate_sensor(mdp2, x1_count, 0.95)
#    print(objectValue - v2)

def main_2():
    num_att = 2
    h = 2
    m = -100
    M = 100
    gridworld1 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    gridworld2 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    r_i_1 = [15, 12, 12]
    r_i_2 = [12, 15, 15]
    r_i_list = [r_i_1, r_i_2]
    v1, x1, v_spec_1 = sub_solver(h, m, M, gridworld1, r_i_1)
    v2, x2, v_spec_2 = sub_solver(h, m, M, gridworld2, r_i_2)
    print(gridworld1.sensor_place(x1))
    print(gridworld2.sensor_place(x2))
    mdplist = [gridworld1, gridworld2]
    vlist = [v1, v2]
    regret, x_regret = LP(num_att, h, m, M, mdplist, vlist, r_i_list)
    print(gridworld1.sensor_place(x_regret))
    start_time = time.time()
    objectiveValue, V_spec = evaluate_sensor(gridworld1, x2, r_i_1)
    end_time = time.time()
    return end_time - start_time
def main_3():
    num_att = 3
    h = 2
    m = -100
    M = 100
    gridworld1 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    gridworld2 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    gridworld3 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    r_i_1 = [15, 12, 12]
    r_i_2 = [12, 15, 15]
    r_i_3 = [12, 12, 15]
    r_i_list = [r_i_1, r_i_2, r_i_3]
    v1, x1, v_spec_1 = sub_solver(h, m, M, gridworld1, r_i_1)
    v2, x2, v_spec_2 = sub_solver(h, m, M, gridworld2, r_i_2)
    v3, x3, v_spec_3 = sub_solver(h, m, M, gridworld3, r_i_3)
    # print(gridworld1.sensor_place(x1))
    # print(gridworld2.sensor_place(x2))
    mdplist = [gridworld1, gridworld2, gridworld3]
    vlist = [v1, v2, v3]
    start_time = time.time()
    regret, x_regret = LP(num_att, h, m, M, mdplist, vlist, r_i_list)
    end_time = time.time()
    return end_time - start_time

def main_4():
    num_att = 4
    h = 2
    m = -100
    M = 100
    gridworld1 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    gridworld2 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    gridworld3 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    gridworld4 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    r_i_1 = [15, 12, 12]
    r_i_2 = [12, 15, 15]
    r_i_3 = [15, 12, 15]
    r_i_4 = [12, 12, 15]
    r_i_list = [r_i_1, r_i_2, r_i_3, r_i_4]
    v1, x1, v_spec_1 = sub_solver(h, m, M, gridworld1, r_i_1)
    v2, x2, v_spec_2 = sub_solver(h, m, M, gridworld2, r_i_2)
    v3, x3, v_spec_3 = sub_solver(h, m, M, gridworld3, r_i_3)
    v4, x4, v_spec_4 = sub_solver(h, m, M, gridworld3, r_i_4)
    # print(gridworld1.sensor_place(x1))
    # print(gridworld2.sensor_place(x2))
    mdplist = [gridworld1, gridworld2, gridworld3, gridworld4]
    vlist = [v1, v2, v3, v4]
    start_time = time.time()
    regret, x_regret = LP(num_att, h, m, M, mdplist, vlist, r_i_list)
    end_time = time.time()
    return end_time - start_time
if __name__ == "__main__":
    time1 = main_2()
    # time2 = main_3()
    # time3 = main_4()
    print(time1)
    # print(time2)
    # print(time3)
    
    