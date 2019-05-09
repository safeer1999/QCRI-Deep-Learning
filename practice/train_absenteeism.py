import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np 
from tensorflow import set_random_seed
import os
import pandas as pd

from autoencoders import AutoEncoder

'''np.random.seed(2)
set_random_seed(2)
r = lambda: np.random.randint(1, 3)
x = np.array([[r(),r(),r()] for _ in range(1000)])'''

df = pd.read_csv('/home/safeer/Documents/QCRI/datasets/Absenteeism_at_work.csv', sep = ';')

df = df.iloc[:,5:]

X = df.iloc[:,:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#X_test = sc.transform(X_test)

print(X)
obj = AutoEncoder(encoding_dim= 2, dataset = X)
obj.encoder_decoder()
obj.fit(batch_size = 10, epochs =500)
obj.save()
