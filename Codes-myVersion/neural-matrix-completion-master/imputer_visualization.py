#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:35:29 2019

@author: safeer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as scp

R_complete = np.load("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/R_norm_complete_matrix.npy")
recons_complete = np.load("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/recons_complete_matrix.npy")


mask = scp.load_npz("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/data/MovieLens100K/train_mask.npz").todense()

R_complete = np.add(R_complete, 2.5)
recons_complete = np.add(recons_complete , 2.5)

def inverse(a) :
    return not a

vectorize = np.vectorize(inverse)

mask_inverse = vectorize(mask)

impute_gap = np.multiply(R_complete, mask_inverse)

fill_gap = np.multiply(recons_complete, mask)

imputed_val = np.add(impute_gap,fill_gap)


imputed_val = np.squeeze(np.asarray(imputed_val.flatten()))

x_axis = np.arange(imputed_val.shape[0])
y_axis = imputed_val.flatten()

y_axis = R_complete.flatten()

plt.scatter(x_axis, y_axis, s = 0.5)
plt.show()


