#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 01:13:52 2022

@author: cuifuheng
"""

# %load generate_GMM.py
import numpy as np
import scipy as sp
import copy as copy

n_iter=2 #Times of experiments

def main(seed, modes):
    np.random.seed(seed)

    #Define GMM parameters
    K = 3        # number of clusters
    D = 1        ###dimension
    a=1 #hyperparameter for Dirichlet distr'n for Pi
    Pi = np.random.dirichlet((a,a,a))
    zvalues = np.array([0,1,2])
  
    if modes == 'sep':
        #MEANvalues = np.array([[0,0],
        #                       [2,2],
        #                       [4,4]])                   ### mean values
        MEANvalues = np.array([0,2,4])
    elif modes == 'insep':
        #MEANvalues = np.array([[0,0],
        #                       [1,1],
        #                       [2,2]])                   ### mean values
        MEANvalues = np.array([0,1,2])
    #sigvalues = np.array([[[1,0],
    #                       [0,1]],
    #                      [[1,0],
    #                       [0,1]],
    #                      [[1,0],
    #                       [0,1]]]) # deviance used in sampling layer
    
    sigvalues=np.array([1,1,1])
    
    #SIG=np.array([[1,0],
    #              [0,1]]) # deviance used in top layer (sampling mu_k_set)
    SIG=0.3
    
    #Sample training z and y
    #set two different datasets i.e. K_set=2
    #row: set; columns: different indicators or variables
    
    K_set=2
    meanvalues=np.zeros((K_set,K,D))
    
    
    N = 1000
    z = np.zeros((K_set,N),dtype = 'int')
    y = np.zeros((K_set,N,D))
    
    #import pdb
    #pdb.set_trace()
    
    for i in range(K_set):
        meanvalues[i]=np.array([np.random.normal(MEANvalues[0],SIG),np.random.normal(MEANvalues[1],SIG),\
                                np.random.normal(MEANvalues[2],SIG)]).reshape((K,D))
        
        
        
        zrng = np.random.rand(K_set*N).reshape((K_set,N))
        yrng = np.random.randn(K_set*N*D).reshape((K_set,N,D))
        
        for j in range(N):
            ind = np.random.multinomial(1, Pi).argmax()
            z[i,j]= zvalues[ind]
            if D==1:
                y[i,j,] = sigvalues[ind]*yrng[i,j,] + meanvalues[i,ind,]
            else:
                y[i,j,] = sigvalues[ind]@yrng[i,j,] + meanvalues[i,ind,]
            
            
        #y = y.reshape(N,D)

    #Concatenate and save training
    gmm_data = {'y': y, 'z': z, 'N':N,'K': K,'D':D, 'Pi' : Pi, 'zvals': zvalues, 'meanvals': meanvalues, 'sigvals':sigvalues}
    np.save('./sim_data/gmm_data_{}_seed{}'.format(modes,seed),gmm_data)


    #Sample test z and y
    
    N_test = 1000
    z_test = np.zeros((K_set,N_test),dtype = 'int')
    y_test = np.zeros((K_set,N_test,D))
    for i in range(K_set):
        meanvalues[i]=np.array([np.random.normal(MEANvalues[0],SIG),np.random.normal(MEANvalues[1],SIG),\
                                np.random.normal(MEANvalues[2],SIG)]).reshape((K,D))
        
        
        zrng = np.random.rand(K_set*N_test).reshape((K_set,N_test))
        yrng = np.random.randn(K_set*N_test*D).reshape((K_set,N_test,D))
        
        for j in range(N_test):
            ind = np.random.multinomial(1, Pi).argmax()
            z_test[i,j]= zvalues[ind]
            if D==1:
                y_test[i,j,] = sigvalues[ind]*yrng[i,j,] + meanvalues[i,ind,]
            else:
                y_test[i,j,] = sigvalues[ind]@yrng[i,j,] + meanvalues[i,ind,]
            
            
        #y_test = y_test.reshape(N_test,D)

    

    #Concatenate and save test data
    gmm_data_test = {'y': y_test, 'z': z_test, 'N':N_test,'K': K,'D':D, 'Pi' : Pi, 'zvals': zvalues, 'meanvals': meanvalues, 'sigvals':sigvalues}
    np.save('./sim_data/gmm_data_test_{}_seed{}'.format(modes,seed),gmm_data_test)

for i in range(n_iter):
    seed = 100+i
    main(seed,'sep')