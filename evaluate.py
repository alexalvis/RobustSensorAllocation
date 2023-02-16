import numpy as np
import GridWorldV2
import sensorallocation
import sensorallocationGeneral2

def EvalConstraint(mdp, pi1, pi2, v2):
    r_d = -100 #Defender's reward
    r_i = 100 #Attacker's reward
    st_len = len(mdp.statespace)
    act_len_att = len(mdp.A)
    act_len_def = 2
    Z = 100000
    R_d = sensorallocationGeneral2.AssignReward(mdp.G, mdp.statespace, act_len_def, act_len_att, r_d)
    R_i = sensorallocationGeneral2.AssignReward(mdp.G, mdp.statespace, act_len_def, act_len_att, r_i)
    v1 = np.zeros(st_len)
    for i in range(st_len):
        v1[i] = -v2[i]
    P = sensorallocationGeneral2.transfer_P(mdp)
    for i in range(st_len):
        if sum(pi1[i]) != 1:
            print("Policy 1 Error")
            
        if sum(pi2[i]) != 1:
            print("Policy 2 Error")
    
    w1 = np.zeros((st_len, act_len_def, act_len_att))
    w2 = np.zeros((st_len, act_len_def, act_len_att))
    for i in range(st_len):
        for j in range(act_len_def):
            for k in range(act_len_att):
                if pi1[i][j] == 0:
                    w1[i][j][k] = 0
                    w2[i][j][k] = 0
                else:
                    w1[i][j][k] = calSum(P, i, j, k, v1)
                    w2[i][j][k] = calSum(P, i, j, k, v2)
    
    for i in range(st_len):
        for j in range(act_len_att):
            if pi2[i][j] == 1:
                calUti_d = calUti(R_d, pi1, w1, i, j)
                calUti_i = calUti(R_i, pi1, w2, i, j)
                print("state is: " + str(i) + " action taken is " + str(j) + " difference is: " + str(v1[i] - calUti_d))
                # if v1[i] - calUti_d > 0:
                    # print("state " + str(i) + " U1 error " + str(v1[i] - calUti_d))
                # if v2[i] != calUti_i:
                #     print("state " + str(i) + " U2 error " + str(v2[i] - calUti_i))
            else:
                calUti_d = calUti(R_d, pi1, w1, i, j)
                calUti_i = calUti(R_i, pi1, w2, i, j)
                print("state is: " + str(i) + " action not taken is " + str(j) + " difference is: " + str(v1[i] - calUti_d))
                # if v1[i] - calUti_d > Z:
                #     print("violation on U1 state: " + str(i))
                # if v2[i] - calUti_i < 0 or v2[i] - calUti_i > Z:
                #     print("violation on U2 state: " + str(i))
    print(w1[9],w2[9])
                
            
    
def calSum(P, i, j, k, v):
    temp = 0
    for index in range(len(P[i][j][k])):
        temp += v[index] * P[i][j][k][index]
    return temp

def calUti(R, pi, w, i, j):
    gamma = 0.95
    temp = 0
    for act_def in range(2):
        temp += pi[i][act_def] * R[i][act_def][j] + gamma * w[i][act_def][j]
    return temp
        
            
def main():
    goallist = [(1, 4)]
    gridworld = GridWorldV2.CreateGridWorld(goallist)
    h = 2 #Number of sensors allocation
    m = -1000
    M = 1000
    v2, x1, v_spec_2 = sensorallocation.sub_solver(h, m, M, gridworld, 100)
    pi1 = np.zeros((len(gridworld.statespace), 2))
    pi2 = np.zeros((len(gridworld.statespace), 4))
    pi2[0][3] = 1
    pi2[1][1] = 1   
    pi2[2][0] = 1
    pi2[3][0] = 1
    pi2[4][3] = 1
    pi2[5][3] = 1
    pi2[6][0] = 1
    pi2[7][0] = 1
    pi2[8][0] = 1
    pi2[9][0] = 1
    pi2[10][3] = 1
    pi2[11][0] = 1
    pi2[12][2] = 1
    pi2[13][2] = 1
    pi2[14][3] = 1
    pi2[15][0] = 1
    pi2[16][0] = 1
    pi2[17][2] = 1
    pi2[18][0] = 1
    pi2[19][0] = 1
    pi2[20][0] = 1
    pi2[21][0] = 1
    pi2[22][2] = 1
    for i in range(len(x1)):
        if x1[i] == 1:
            pi1[i][1] = 1
        else:
            pi1[i][0] = 1
    gridworld.ChangeGoalTrans()
    EvalConstraint(gridworld, pi1, pi2, v_spec_2) 

if __name__ == "__main__":
    main()