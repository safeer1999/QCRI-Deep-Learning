#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:48:22 2019

@author: safeer
"""

import pandas as pd
import numpy as np
import as os
import matplotlib.pyplot as plt


X_R = [] # list of R batches arrays
X_recons = [] # list of recons batches arrays

#Extracting R_batch files
for i in range(1000) :
    
    file_name = str(R_and_recons/") + str("R") + str(i) + str(".npy")
    R = np.load(file_name)
    X_R.append(R.flatten())
    
#Extracting recons_batch files
for i in range(1000) :
    
    file_name = str("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/R_and_recons/") + str("recons") + str(i) + str(".npy")
    recons = np.load(file_name)
    X_recons.append(recons.flatten())

x_axis = np.arange(17000)


os.mkdir("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/Visualizations")

for i  in range(1000) :
    
    y_axis_R = X_R[i]
    y_axis_recons = X_recons[i]
    
    plt.scatter(x_axis,y_axis_R, s= 1, c = "red")
    plt.scatter(x_axis,y_axis_recons,s = 1, c = 'blue')
    
    plt.xlabel("indices")
    plt.ylabel("values")
    #plt.legend("NMC")
    plt.title("Neural Matrix Completion")
    
    
    file_name = str("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/Visualizations/") + str("image") + str(i) + str(".png")
    
    plt.savefig(file_name)
    
    plt.close()
    
    