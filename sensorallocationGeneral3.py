from mip import *
import numpy as np
import GridWorldV2
import sensorallocationGeneral2

#Multi type attacker, minimize regret

def LP_regretMinimize(num_att, h, Z, mdplist, v_i, r_i, r_d):
    model = Model(solver_name=GRB)
    gamma = 0.95
    mdp = mdplist[0]
    st_len = len(mdp.statespace)
    init = mdplist[0].init
    act_len_att = len(mdp.A)
    act_len_def = 2
    r_d_min = []
    r_i_max = []
    for i in range(num_att):
        r_d_min.append(min(r_d[i]))
        r_i_max.append(max(r_i[i]))
    y = model.add_var(var_type=CONTINUOUS)  #y is regret
    pi1 = [[model.add_var(var_type=BINARY) for i in range(act_len_def)] for j in range(st_len)]  # Defender's policy
    pi2 = [[[model.add_var(var_type=BINARY) for i in range(act_len_att)] for j in range(st_len)] for att_type in range(num_att)] # Attacker's policy of type i
    U1 = [[model.add_var(lb=r_d_min[att_type], ub=0, var_type=CONTINUOUS) for i in range(st_len)] for att_type in range(num_att)]  # Defender's utility against type i
    U2 = [[model.add_var(lb=0, ub=r_i_max[att_type], var_type=CONTINUOUS) for i in range(st_len)] for att_type in range(num_att)]  # Attacker's Utility
    w1 = [[[[model.add_var(lb=r_d_min[att_type], ub=0, var_type=CONTINUOUS) for i in range(act_len_att)] for j in range(act_len_def)] for k in
          range(st_len)] for att_type in range(num_att)]  # Defender's replacement
    w2 = [[[[model.add_var(lb=0, ub=r_i_max[att_type], var_type=CONTINUOUS) for i in range(act_len_att)] for j in range(act_len_def)] for k in
          range(st_len)] for att_type in range(num_att)]  # Attacker's replacement
    U = mdp.U

    model.infeas_tol = 1e-3
    # model.integer_tol = 1e-9
    # model.opt_tol = 1e-9
    model.seed = 3
    model.objective = minimize(y)
    #Regret
    for i in range(num_att):
        init = mdplist[i].init
        model += y - (v_i[i] - xsum(init[j] * U1[i][j] for j in range(st_len))) >= 0

    #Sensors can not be placed outside U
    for i in range(st_len):
        if mdp.statespace[i] not in U:
                model += pi1[i][0] == 1
                model += pi1[i][1] == 0

    #Attacker's action constraint
    for att_type in range(num_att):
        for i in range(st_len):
            model += xsum(pi2[att_type][i][j] for j in range(act_len_att)) == 1

    #Defender's action constraint
    for i in range(st_len):
        model += xsum(pi1[i][k] for k in range(act_len_def)) == 1
    # Number of sensors constraint
    model += xsum(pi1[i][1] for i in range(st_len)) <= h

    # Utility Constraint
    for att_type in range(num_att):
        mdp = mdplist[att_type]
        R_d = sensorallocationGeneral2.AssignReward(mdp.G, mdp.statespace, act_len_def, act_len_att, r_d[att_type])
        # print(R_d)
        R_i = sensorallocationGeneral2.AssignReward(mdp.G, mdp.statespace, act_len_def, act_len_att, r_i[att_type])
        # print(R_i)
        for i in range(st_len):
            if mdp.statespace[i] not in mdp.G:
                for act_att in range(act_len_att):
                    model += U1[att_type][i] - xsum(
                        pi1[i][act_def] * R_d[i][act_def][act_att] + gamma * w1[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             <= (1 - pi2[att_type][i][act_att]) * Z

                    model += U1[att_type][i] - xsum(
                        pi1[i][act_def] * R_d[i][act_def][act_att] + gamma * w1[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             >= (1 - pi2[att_type][i][act_att]) * -Z

                    model += U2[att_type][i] - xsum(
                        pi1[i][act_def] * R_i[i][act_def][act_att] + gamma * w2[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             <= (1 - pi2[att_type][i][act_att]) * Z

                    model += U2[att_type][i] - xsum(
                        pi1[i][act_def] * R_i[i][act_def][act_att] + gamma * w2[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             >= 0

    # Replacement of w1 and w2
    for att_type in range(num_att):
        mdp = mdplist[att_type]
        P = sensorallocationGeneral2.transfer_P(mdp)
        for i in range(st_len):
            if mdp.statespace[i] not in mdp.G:
                for act_def in range(act_len_def):
                    for act_att in range(act_len_att):
                        model += w1[att_type][i][act_def][act_att] >= xsum(
                            P[i][act_def][act_att][j] * U1[att_type][j] for j in range(st_len)) - Z * (1 - pi1[i][act_def])
                        model += w1[att_type][i][act_def][act_att] <= xsum(
                            P[i][act_def][act_att][j] * U1[att_type][j] for j in range(st_len)) + Z * (1 - pi1[i][act_def])
                        model += w1[att_type][i][act_def][act_att] >= -Z * pi1[i][act_def]
                        model += w1[att_type][i][act_def][act_att] <= Z * pi1[i][act_def]

                        model += w2[att_type][i][act_def][act_att] >= xsum(
                            P[i][act_def][act_att][j] * U2[att_type][j] for j in range(st_len)) - Z * (1 - pi1[i][act_def])
                        model += w2[att_type][i][act_def][act_att] <= xsum(
                            P[i][act_def][act_att][j] * U2[att_type][j] for j in range(st_len)) + Z * (1 - pi1[i][act_def])
                        model += w2[att_type][i][act_def][act_att] >= -Z * pi1[i][act_def]
                        model += w2[att_type][i][act_def][act_att] <= Z * pi1[i][act_def]
    
    #Utilities at goal states
    for att_type in range(num_att):
        mdp = mdplist[att_type]
        r_d_spec = r_d[att_type]
        r_i_spec = r_i[att_type]
        for i in range(st_len):
            if mdp.statespace[i] in mdp.G:
                model += U1[att_type][i] == r_d_spec[mdp.G.index(mdp.statespace[i])]
                model += U2[att_type][i] == r_i_spec[mdp.G.index(mdp.statespace[i])]
                for act_def in range(act_len_def):
                    for act_att in range(act_len_att):
                        model += w1[att_type][i][act_def][act_att] == r_d_spec[mdp.G.index(mdp.statespace[i])] * pi1[i][act_def]
                        model += w2[att_type][i][act_def][act_att] == r_i_spec[mdp.G.index(mdp.statespace[i])] * pi1[i][act_def]
    
    status = model.optimize()  # Set the maximal calculation time
    print(status)
    if status == OptimizationStatus.OPTIMAL:
        print("The model objective is:", model.objective_value)
        sensorplace = [pi1[i][1].x for i in range(st_len)]
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        for i in range(st_len):
            temp1 += U1[0][i].x * mdplist[0].init[i]
            temp2 += U1[1][i].x * mdplist[1].init[i]
            temp3 += U2[0][i].x * mdplist[0].init[i]
            temp4 += U2[1][i].x * mdplist[1].init[i]
        print(temp1 - v_i[0], temp2 - v_i[1])
        print(temp3, temp4)
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
    else:
        print("The model objective is:", model.objective_value)
    return model.objective_value, sensorplace

def main():
    #5 X 5 gridworld example
    num_att = 2  #Number of attacker types
    h = 2  #Number of sensor constraints
    Z = 10
    goallist1 = [(1, 4)]
    goallist2 = [(4, 4)]
    mdp1 = GridWorldV2.CreateGridWorld(goallist1)
    mdp2 = GridWorldV2.CreateGridWorld(goallist2)
    mdp1.ChangeGoalTrans()
    mdp2.ChangeGoalTrans()
    mdplist = [mdp1, mdp2]
    r_dlist = [[-1], [-0.95]]
    r_ilist = [[1], [0.95]]
    v_ilist = []
    sensorConfiglist = []
    for i in range(num_att):
        v_i, sensor_i = sensorallocationGeneral2.LP(mdplist[i], h, r_dlist[i], r_ilist[i])
        v_ilist.append(v_i)
        sensorConfiglist.append(sensor_i)
    V_regret, sensor_regret = LP_regretMinimize(num_att, h, Z, mdplist, v_ilist, r_ilist, r_dlist)
    print(sensor_regret)
    
def main_v2():
    #8 X 8 gridworld example
    num_att = 2
    h = 2
    Z = 10
    gridworld1 = GridWorldV2.CreateGridWorld_V2([0.3, 0.7])
    gridworld2 = GridWorldV2.CreateGridWorld_V2([0.7, 0.3])
    mdplist = [gridworld1, gridworld2]
    r_d = [-1.5, -1, -1]
    r_i_1 = [1.5, 1.2, 1.5]
    r_i_2 = [1.2, 1.5, 1]
    r_d_list = [r_d, r_d]
    r_i_list = [r_i_1, r_i_2]
    v_i_list = []
    sensorConfiglist = []
    for i in range(num_att):
        v_i, sensor_i = sensorallocationGeneral2.LP(mdplist[i], h, r_d_list[i], r_i_list[i])
        sensor_p = mdplist[i].sensor_place(sensor_i)
        v_i_list.append(v_i)
        sensorConfiglist.append(sensor_p)
    print(v_i_list, sensorConfiglist)
    V_regret, sensor_regret = LP_regretMinimize(num_att, h, Z, mdplist, v_i_list, r_i_list, r_d_list)
    sensor_r = gridworld1.sensor_place(sensor_regret)
    return v_i_list, sensorConfiglist, V_regret, sensor_r

if __name__ == "__main__":
    v_i_list, sensorConfiglist, V_regret, sensor_r = main_v2()
    

