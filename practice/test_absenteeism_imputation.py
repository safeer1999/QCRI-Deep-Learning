import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np 
from tensorflow import set_random_seed
import os
import pandas as pd
from keras.models import load_model
import math

test_df = pd.read_csv('/home/safeer/Documents/QCRI/datasets/AbsenteeismMissingData.csv')

df = pd.read_csv('/home/safeer/Documents/QCRI/datasets/Absenteeism_at_work.csv', sep = ';')

df = df.iloc[:,5:]

X = df.iloc[:,:].values

X_test = test_df.iloc[:,5:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.fit_transform(X_test)

#print(df)

y = []

for i  in range(740) :

	for j in range(16) :

 		if np.isnan(X_test[i,j]) :
 			X_test[i,j] = 0
 			y.append((i,j,X[i,j]))

#print(y)

encoder = load_model(r'./weights/encoder_weights.h5')
decoder = load_model(r'./weights/decoder_weights.h5')

#print(encoder)
#print(decoder)

#print(X)
#print(X_test)

deconstructed_X = encoder.predict(X)
print(deconstructed_X)
reconstructed_X = decoder.predict(deconstructed_X)

print(reconstructed_X)

from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(reconstructed_X,X_test))

print("RMSE: ",rmse) 


