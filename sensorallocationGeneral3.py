from mip import *
import numpy as np
import GridWorldV2
import sensorallocationGeneral2

#Multi type attacker, minimize regret

def LP_regretMinimize(num_att, h, Z, mdplist, v_i, r_i, r_d):
    model = Model(solver_name=GRB)
    gamma = 0.95
    mdp = mdplist[0]
    stlen = len(mdp.statespace)
    init = mdplist[0].init
    act_len_att = len(mdp.A)
    act_len_def = 2
    y = model.add_var()  #y is regret
    pi1 = [[model.add_var(var_type=BINARY) for i in range(act_len_def)] for j in range(st_len)]  # Defender's policy
    pi2 = [[[model.add_var(var_type=BINARY) for i in range(act_len_att)] for j in range(st_len)] for att_type in range(num_att)] # Attacker's policy of type i
    U1 = [[model.add_var(lb=r_d, ub=0) for i in range(st_len)] for att_type in range(num_att)]  # Defender's utility against type i
    U2 = [[model.add_var(lb=0, ub=r_i) for i in range(st_len)] for att_type in range(num_att)]  # Attacker's Utility
    w1 = [[[[model.add_var(lb=r_d, ub=0) for i in range(act_len_att)] for j in range(act_len_def)] for k in
          range(st_len)] for att_type in range(num_att)]  # Defender's replacement
    w2 = [[[[model.add_var() for i in range(act_len_att)] for j in range(act_len_def)] for k in
          range(st_len)] for att_type in range(num_att)]  # Attacker's replacement
    U = mdp.U

    model.infeas_tol = 1e-3
    model.objective = minimize(y)
    #Regret
    for i in range(num_att):
        model += y >= (xsum(init[j] * U1[i][j] for j in range(stlen)) - v_i[i])

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
        for i in range(st_len):
            if mdp.statespace[i] not in mdp.G:
                for act_att in range(act_len_att):
                    model += U1[att_type][i] - xsum(
                        pi1[i][act_def] * R_d[att_type][i][act_def][act_att] + gamma * w1[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             <= (1 - pi2[att_type][i][act_att]) * Z

                    model += U1[att_type][i] - xsum(
                        pi1[i][act_def] * R_d[att_type][i][act_def][act_att] + gamma * w1[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             >= (1 - pi2[att_type][i][act_att]) * -Z

                    model += U2[att_type][i] - xsum(
                        pi1[i][act_def] * R_i[att_type][i][act_def][act_att] + gamma * w2[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             <= (1 - pi2[att_type][i][act_att]) * Z

                    model += U2[att_type][i] - xsum(
                        pi1[i][act_def] * R_i[att_type][i][act_def][act_att] + gamma * w2[att_type][i][act_def][act_att] for act_def in range(act_len_def)) \
                             >= 0

    # Replacement of w1 and w2
    for att_type in range(num_att):
        mdp = mdplist[att_type]
        P = transfer_P(mdp)
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




