# -*- coding: utf-8 -*-
#Gridworld use for robust sensor allocation
import numpy as np
import copy
import sys

class GridWorld:
    def __init__(self, width, height, stoPar):
        self.width = width
        self.height = height
        self.stoPar = stoPar
        self.A = [(0, 1), (0, -1), (-1, 0), (1, 0)] #E, W, S, N
        self.complementA = self.getComplementA()
        self.statespace = self.getstate()
        self.gettrans()
        self.F = []
        self.G = []
        self.IDS = []
        self.U = []

    def getstate(self):
        statespace = []
        for i in range(self.width):
            for j in range(self.height):
                statespace.append((i, j))
        return statespace
    
    def checkinside(self, st):
        if st in self.statespace:
            return True
        return False
    
    def getComplementA(self):
        complementA = {}
        complementA[(0, 1)] = [(1, 0), (-1, 0)]
        complementA[(0, -1)] = [(1, 0), (-1, 0)]
        complementA[(1, 0)] = [(0, 1), (0, -1)]
        complementA[(-1, 0)] = [(0, 1), (0, -1)]
        return complementA
        
    def gettrans(self):
        #Calculate transition
        stoPar = self.stoPar
        trans = {}
        for st in self.statespace:
            trans[st] = {}
            for act in self.A:
                trans[st][act] = {}
                trans[st][act][st] = 0
                tempst = tuple(np.array(st) + np.array(act))
                if self.checkinside(tempst):
                    trans[st][act][tempst] = 1 - 2*stoPar
                else:
                    trans[st][act][st] += 1- 2*stoPar
                for act_ in self.complementA[act]:
                    tempst_ = tuple(np.array(st) + np.array(act_))
                    if self.checkinside(tempst_):
                        trans[st][act][tempst_] = stoPar
                    else:
                        trans[st][act][st] += stoPar
        self.stotrans = trans
        flag = self.checktrans()
        if not flag:
            sys.exit("Transition is incorrect, quit")
    
    def st2st(self):
        st2st = []
        for st in self.statespace:
            for act in self.A:
                for st_ in self.stotrans[st][act].keys():
                    st2st.append((st, st_))
        self.st2st = st2st

    
    def checktrans(self):
        for st in self.statespace:
            for act in self.A:
                if abs(sum(self.stotrans[st][act].values())-1) > 0.01:
                    print("st is:", st, " act is:", act, " sum is:", sum(self.stotrans[st][act].values()))
                    return False
        print("Transition is correct")
        return True

    
    def addGoal(self, goallist):
        #Add true Goals
        for st in goallist:
            self.G.append(st)
            for act in self.A:
                self.stotrans[st][act] = {}
                self.stotrans[st][act]["Sink"] = 1.0

        
    def addBarrier(self, Barrierlist):
        #Add barriers in the world
        #If we want to add barriers, Add barriers first, then calculate trans, add True goal, add Fake goal, add IDS
        for st in Barrierlist:
            self.statespace.remove(st)
            
    def addU(self, Ulist):
        for st in self.statespace:
            if st not in Ulist:
                self.U.append(st)
        
    


def createGridWorldBarrier_new2():
    gridworld = GridWorld(6, 6, 0.1)
    goallist = [(3, 4), (5, 0)] 
#    barrierlist = [(0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (2, 2), (4, 2)]
    barrierlist = []
    gridworld.addBarrier(barrierlist)
    fakelist = [(1, 4), (4, 5)] #case 1
    # fakelist = [(0, 2), (5, 3)] #case 2
    IDSlist = [(0, 4), (1, 2), (2, 3), (3, 3), (5, 4)]
#    IDSlist = [(6, 5), (4, 5)]
#    fakelist = [(4, 6), (7, 4)]
    Ulist = []  #This U is the states that can place sensors
    for i in range(6):
        for j in range(2, 4):
            Ulist.append((i, j))
    gridworld.addU(Ulist)
    gridworld.gettrans()
    gridworld.addFake(fakelist)
    gridworld.addGoal(goallist)
    gridworld.addIDS(IDSlist)
#    V_0 = gridworld.init_preferred_attack_value()
    # reward = gridworld.getreward_def(1)   #Cant use this as the initial reward
    # reward = gridworld.initial_reward()
    reward = gridworld.initial_reward_withoutDecoy()
    reward = gridworld.initial_reward_manual([2.016, 1.826])
    print(reward)
#    print(reward)
#    policy, V = gridworld.getpolicy(reward)
    policy, V = gridworld.getpolicy_det(reward)
#    policy = gridworld.randomPolicy()
    reward_d = gridworld.getreward_def(1)
    # print(reward_d)
    print(V)
    V_def = gridworld.policy_evaluation(policy, reward_d)
    return gridworld, V_def, policy    


    
    
if __name__ == "__main__":
#    gridworld, V, policy = createGridWorld()
    gridworld, V_def, policy = createGridWorldBarrier_new3()
    Z = gridworld.stVisitFre(policy)
    Z_act = gridworld.stactVisitFre(policy)
#    print(V_def[14], Z[20], Z[48])
#    print(Z[35], Z[54])