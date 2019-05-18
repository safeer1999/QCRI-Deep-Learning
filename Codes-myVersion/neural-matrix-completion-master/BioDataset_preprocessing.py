import pandas as pd
import numpy as np    
import random
import sys
import scipy.sparse as scp

#df = pd.read_csv("/home/safeer/Documents/Bio_Dataset/GROUND_TRUTH1200.txt",sep = " ") #Complete Dataset
df_train_mask = pd.read_csv("/home/safeer/Documents/Bio_Dataset/DATA1200.txt",sep = " ") # training mask values

#adding columns
columns = [str("snip"+str(i)) for i in range(8955)]
#df.columns = columns
df_train_mask.columns = columns

#replacing missing values with -1
df_train_mask.replace('.', -1, inplace = True)
#df = df.iloc[:,1:]
df_train_mask = df_train_mask.iloc[:,1:]
#print(df)

#type casting from string to int
#X = df.iloc[:,:].values
#X   = X.astype('int')
X_mask = df_train_mask.iloc[:,:].values.astype('int')


#adding all elements by one to ensure representation of 0 is missing value and not a input value
#X = np.add(X,1)
X_mask = np.add(X_mask,1)
#print(X)


#creating the mask
X_mask = np.where(X_mask == 0, True, False)

#np.set_printoptions(threshold = np.inf)
#print(X_mask)

#creating the validation mask
#X_val_mask = np.random.binomial(1,0.15,size= X.shape)

#converting the matrix to sparse matrix
#the dataloader class loads the matrix from the .npz file as a sparse matrix
#sparse_X = scp.csc_matrix(X)
sparse_X_mask = scp.csc_matrix(X_mask)
#sparse_X_val_mask = scp.csc_matrix(X_val_mask)




#scp.save_npz("./Bio_Dataset/rating",sparse_X)
scp.save_npz("./Bio_Dataset/train_mask",sparse_X_mask)
#scp.save_npz("./Bio_Dataset/val_mask",sparse_X_val_mask)