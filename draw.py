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
    
def draw_att():
    x = [1, 2, 3, 4]
    y1 = [6.7431, 5.9604, 5.2428, 0]
    y2 = [7.4487, 6.5584, 5.7437, 0]
    y1_regret = [6.9624, 6.1235, 5.2476, 0]
    y2_regret = [7.6680, 7.0389, 6.5138, 0]
    
    plt.plot(x, y1, marker = "*", markersize = 8, linewidth = 4, color = 'blue', label = "Optimal policy against attacker 1")
    plt.plot(x, y2, marker = "*", markersize = 8, linewidth = 4, color = 'red', label = "Optimal policy against attacker 2")
    plt.plot(x, y1_regret, marker = "*", markersize = 8, linewidth = 4, color = 'black', label = "Regret minimization policy against attacker 1")
    plt.plot(x, y2_regret, marker = "*", markersize = 8, linewidth = 4, color = 'green', label = "Regret minimization policy against attacker 2")
    
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Number of sensors", fontsize = 16)
    plt.yticks([8, 7, 6, 5, 4, 3, 2, 1, 0])
    plt.ylabel("Attacker's value", fontsize = 16) 
    plt.legend()
    plt.show()
    
def draw_bar_plot():

 
    # create a dataset
    y = [5.9604, 6.1235, 7.0826, 6.5584, 7.0389, 7.4498]
    x = ('A', 'B', 'C', 'D', 'E', 'F')
    # x = [0, 1, 2, 3, 4, 5, 6, 7]
    # y = [160, 167, 17, 130, 120, 40, 105, 70]
    fig, ax = plt.subplots()
    width = 0.5
    ind = np.arange(len(x))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel("Attacker's value", fontsize = 16)
    plt.xlim(0, 9)
    plt.yticks(ind, ['Attacker 1 value \n against $\pi_1$', 'Attacker 1 value \n against $\pi_w$', 'Attacker 1 value \n against $\pi_2$', 'Attacker 2 value \n against $\pi_2$', 'Attacker 2 value \n against $\pi_w$', 'Attacker 2 value \n against $\pi_1$'], fontsize = 12)
    ax.barh(ind, y, width, color=['blue', 'black', 'cyan', 'red', 'green', 'orange'])
 
    for i, v in enumerate(y):
        ax.text(v+0.35 , i , str(v),
            color = 'blue', fontweight = 'bold')
    plt.show()
    
if __name__ == "__main__":
    # draw()
    # draw_att()
    draw_bar_plot()