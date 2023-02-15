import numpy as np

class World():
    def __init__(self):
        self.statespace = [0, 1, 2, 3]
        self.A = ["a", "b"]
        self.stotrans = self.definetrans()

    def definetrans(self):
        P = {}
        for i in self.statespace:
            P[i] = {}
            for j in self.A:
                P[i][j] = {}
                for k in self.statespace:
                    P[i][j][k] = 0
        P[0]["a"][1] = 1
        P[0]["b"][2] = 1
        P[1]["a"][3] = 1
        P[1]["b"][1] = 1
        P[2]["a"][2] = 1
        P[2]["b"][3] = 1

