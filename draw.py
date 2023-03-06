# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:26:31 2023

@author: 53055
"""

import numpy as np
import matplotlib.pyplot as plt

def draw():
    x = [1, 2, 3, 4]
    y1 = [-6.3697, -4.9677, -4.3673, 0]
    y2 = [-4.9726, -4.3764, -4.1451, 0]
    y1_regret = [-6.6322, -6.1601, -4.4285, 0]
    y2_regret = [-4.9759, -5.0402, -4.3436, 0]
    
    plt.plot(x, y1, marker = "*", markersize = 8, linewidth = 4, color = 'blue', label = "Optimal policy against attacker 1")
    plt.plot(x, y2, marker = "*", markersize = 8, linewidth = 4, color = 'red', label = "Optimal policy against attacker 2")
    plt.plot(x, y1_regret, marker = "*", markersize = 8, linewidth = 4, color = 'black', label = "Regret minimization policy against attacker 1")
    plt.plot(x, y2_regret, marker = "*", markersize = 8, linewidth = 4, color = 'green', label = "Regret minimization policy against attacker 2")
    
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Number of sensors", fontsize = 16)
    plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0])
    plt.ylabel("Defender's value", fontsize = 16) 
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    draw()