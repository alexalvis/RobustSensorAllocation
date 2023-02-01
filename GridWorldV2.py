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
        else:
            self.stactst()
    
    def stactst(self):
        st2st = []
        for st in self.statespace:
            for act in self.A:
                for st_ in self.stotrans[st][act].keys():
                    if (st, act, st_) not in st2st:
                        st2st.append((st, act, st_))
        self.stast = st2st

    
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
                self.stotrans[st][act][st] = 1.0
        self.stactst()

        
    def addBarrier(self, Barrierlist):
        #Add barriers in the world
        #If we want to add barriers, Add barriers first, then calculate trans, add True goal, add Fake goal, add IDS
        for st in Barrierlist:
            self.statespace.remove(st)
            
    def addU(self, Ulist):
        for st in Ulist:
            if st in self.statespace:
                self.U.append(st)
        

def CreateGridWorld(goallist):
    gridworld = GridWorld(5, 5, 0.1)
    # goallist = [(1, 4)]
    barrierlist = [(0, 2), (2, 3), (3, 1)]
    gridworld.addBarrier(barrierlist)
    Ulist = []
    for i in range(5):
        for j in range(1,3):
            Ulist.append((i, j))
    gridworld.addU(Ulist)
    gridworld.gettrans()
    gridworld.addGoal(goallist)
    return gridworld
   
    
    
if __name__ == "__main__":
    goallist = [(1, 4)]
    gridworld = CreateGridWorld(goallist)