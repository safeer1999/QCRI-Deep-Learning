import os
import tensorflow as tf
import pandas as pd
import time
from autoencoder import *
import sys
import random
import numpy as np


def get_next_batch(dataset, batch_size,step,ind):

    start=step*batch_size
    end=((step+1)*batch_size)
    sel_ind=ind[start:end]
    newdataset=dataset.iloc[sel_ind,:]
  
    return newdataset

    
def calculate_nrmse_loss(reconstructed,input_shape):
    
   original = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='original')
   missing_mask=tf.placeholder(tf.float32,
                                 input_shape,
                                 name='original')
   
   reconstructed_masked_value=tf.multiply(reconstructed, missing_mask)
   original_maksed_value=tf.multiply(original, missing_mask)
   rmse= tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(reconstructed_masked_value,
                                                 original_maksed_value)),axis=0)))
  
   return original, rmse, missing_mask




def train(nonmissing_perc,dataset_train,dataset_test,autoencoder_fun, restore=False,sav=True,checkpoint_file='default.ckpt'):
    input_image, reconstructed_image = autoencoder_fun(batch_shape)
    original, loss, missing_mask = calculate_nrmse_loss(reconstructed_image,[batch_size,feature_size])
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    start=time.time()
    loss_val_list_train=0
    loss_val_list_test=0
    with tf.Session() as session:
        
            session.run(init)
            dataset_size_train = dataset_train.shape[0]
            dataset_size_test = dataset_test.shape[0]
            print("Dataset size for training:", dataset_size_train)
            print("Dataset size for validation:", dataset_size_test)
            num_iters = (num_epochs * dataset_size_train)//batch_size
            print("Num iters:", num_iters)
            ind_train=0
            for i in range(num_epochs):
                ind_train= np.append(ind_train, np.random.permutation(np.arange(dataset_size_train)))
            ind_test=0
            iters=num_epochs*dataset_size_train//dataset_size_test+1
            for i in range(iters):
                ind_test= np.append(ind_test, np.random.permutation(np.arange(dataset_size_test)))
        
            for step in range(num_iters):
                temp = get_next_batch(dataset_train, batch_size,step,ind_train)
                train_batch=np.asarray(temp).astype("float32")
                frac=nonmissing_perc
                sample = np.random.binomial(1, frac, size=temp.shape[0]*temp.shape[1])
                sample2=sample.reshape(temp.shape[0],temp.shape[1])
                missing_ones=np.ones_like(sample2) - sample2
                corrupted=temp*sample2
                corrupted_batch=np.asarray(corrupted).astype("float32")       
                train_loss_val,_= session.run([loss,optimizer], 
                                   feed_dict={input_image: corrupted_batch, original:train_batch,missing_mask:missing_ones})              
                loss_val_list_train=np.append(loss_val_list_train,train_loss_val)
                
                
                
                temp = get_next_batch(dataset_test, batch_size,step, ind_test)
                test_batch=np.asarray(temp).astype("float32")
                frac=nonmissing_perc
                sample = np.random.binomial(1, frac, size=temp.shape[0]*temp.shape[1])
                sample2=sample.reshape(temp.shape[0],temp.shape[1])
                missing_ones=np.ones_like(sample2) - sample2
                corrupted=temp*sample2
                corrupted_batch=np.asarray(corrupted).astype("float32")  
                
                test_loss_val = session.run(loss, 
                                       feed_dict={input_image: corrupted_batch, original:test_batch,missing_mask:missing_ones })
                loss_val_list_test=np.append(loss_val_list_test,test_loss_val)
                
                if step%30==0:
                    print(step, "/", num_iters, train_loss_val,test_loss_val)             
            if sav:
                save_path = saver.save(session, checkpoint_file)
                print(("Model saved in file: %s" % save_path))
            
    end=time.time()
    el=end-start
    print(("Time elapsed %f" %el))
    return(loss_val_list_train, loss_val_list_test)




#[3]###################################
#tf.reset_default_graph()
#with tf.Graph().as_default(): 
if __name__ == '__main__':

        input_name='/home/safeer/Documents/QCRI/datasets/AbsenteeismMissingData.csv'# data set
        output_path='imputationmodel.ckpt' #'imputationmodel.ckpt'
        feature_size=20 #Dimension of the feature, 17176
        nonmissing_perc=0.7 #Percent of non-missing elements in the data, 0.7
        batch_size=20 #128
        learning_rate = 0.1 #0.1
        num_epochs = 450 #450
		
        df = pd.read_csv(input_name)  
        #print(df,"\n\n\n") 
        df = df.iloc[:,1:]

        #random generation of missing values for testing purposes
        '''random.seed(10)
        rows = list(range(df.shape[0]))
        columns = list(range(df.shape[1]))

        
        for i in range(df.shape[0]) :

            y = random.choice(columns)

            while df.iloc[i,y] == None :
                    y = random.choice(columns)
            

            print(i,y,df.iloc[i,y])
            df.iloc[i,y] = None
                    
                    #df.iloc[i,y] = None


        print(df)
        df.to_csv("AbsenteeismMissingData.csv")'''

        non_missing_values =(df.count(axis=1).sum())
        nonmissing_perc = non_missing_values/(df.shape[0]*df.shape[1])
        print("size = ",(df.shape[0]*df.shape[1]))
        print("non_missing percent: ", nonmissing_perc)

        ####Create set for training & validation, and for testing
        arr=list(range(df.shape[0]))
        random.seed(1)
        random.shuffle(arr)
        train_ind=arr[0:int(df.shape[0]*0.75)]
        test_ind=arr[int(df.shape[0]*0.75):len(arr)]
        df_train = df.iloc[train_ind]
        df_test = df.iloc[test_ind]
      
       
        ########       
        arr=list(range(df_train.shape[0]))
        random.seed(1)
        random.shuffle(arr)
        train_ind=arr[0:int(df_train.shape[0]*0.8)]
        test_ind=arr[int(df_train.shape[0]*0.8):len(arr)]
        dataset_train = df_train.iloc[train_ind]
        dataset_test = df_train.iloc[test_ind]
        
        
        batch_shape = (batch_size, feature_size)
        np.set_printoptions(threshold=np.inf)
        #print(df,"\n\n\n")

        #print(dataset)

        tf.reset_default_graph()
        loss_val_list_train, loss_val_list_test=train(nonmissing_perc,dataset_train,dataset_test,autoencoder_fun=autoencoder4_d, sav=True,restore=False, checkpoint_file=output_path)  
        #np.savetxt("trainloss.csv", loss_val_list_train, delimiter="\t")
        #np.savetxt("validationloss.csv", loss_val_list_test, delimiter="\t")  
        
        
        
  
