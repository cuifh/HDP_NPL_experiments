#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 01:21:15 2022

@author: cuifuheng
"""

# %load evaluate_posterior_toygmm_HDP.py
"""
Evaluate posterior samples predictive performance/time

"""
############################################################
n_iter=1


import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import importlib
from npl.evaluate import gmm_ll as gll

def load_data(type,seed):
    #load test data
    gmm_data_test = np.load('./sim_data/gmm_data_test_{}_seed{}.npy'.format(type,seed),allow_pickle = True).item()

    #Extract parameters from data
    N_test = gmm_data_test['N']
    K = gmm_data_test['K']
    D = gmm_data_test['D']
    y_test = gmm_data_test['y'].reshape((2,N_test,D))
    
    return y_test,N_test,K,D


def load_posterior(method,type,seed,K):
    if method == 'RRNPL':
        #par = pd.read_pickle('./parameters/par_bb_{}_rr_rep{}_B{}_seed{}'.format(type,10,2000,seed))
        par = pd.read_pickle('./parameters/par_bb_{}_rr_rep{}_B{}_seed{}'.format(type,10,100,seed))
        pi_1 =np.array(par['pi_1'])
        mu_1 =np.array(par['mu_1'])
        sigma_1 = np.array(par[['sigma_1']][0])
        pi_2 =np.array(par['pi_2'])
        mu_2 =np.array(par['mu_2'])
        sigma_2 = np.array(par[['sigma_2']][0])
        time = par['time']

    elif method =='FINPL':
        #par = pd.read_pickle('./parameters/par_bb_{}_fi__B{}_seed{}'.format(type,2000,seed))
        par = pd.read_pickle('./parameters/par_bb_{}_fi__B{}_seed{}'.format(type,100,seed))
        pi_1 =np.array(par['pi_1'])
        mu_1 =np.array(par['mu_1'])
        sigma_1 = np.array(par[['sigma_1']][0])
        pi_2 =np.array(par['pi_2'])
        mu_2 =np.array(par['mu_2'])
        sigma_2 = np.array(par[['sigma_2']][0])
        time = par['time']

    elif method == 'NUTS':
        D = 1
        par = pd.read_pickle('./parameters/par_nuts_{}_seed{}'.format(type,seed))
        pi =par.iloc[:,3:K+3].values
        mu =par.iloc[:,3+K: 3+(K*(D+1))].values.reshape(2000,D,K).transpose(0,2,1)
        sigma = par.iloc[:,3+K*(D+1) :3+ K*(2*D+1)].values.reshape(2000,D,K).transpose(0,2,1)
        time = np.load('./parameters/time_nuts_{}_seed{}.npy'.format(type,seed),allow_pickle = True)

    elif method == 'ADVI':
        D = 1
        par = pd.read_pickle('./parameters/par_advi_{}_seed{}'.format(type,seed))
        pi =par.iloc[:,0:K].values
        mu =par.iloc[:,K: (K*(D+1))].values.reshape(2000,D,K).transpose(0,2,1)
        sigma = par.iloc[:,K*(D+1) : K*(2*D+1)].values.reshape(2000,D,K).transpose(0,2,1)
        time = np.load('./parameters/time_advi_{}_seed{}.npy'.format(type,seed),allow_pickle = True)


    return pi_1,mu_1,sigma_1,pi_2,mu_2,sigma_2,time

def eval(method,type):
    ll_test_1 = np.zeros(n_iter)
    ll_test_2 = np.zeros(n_iter)
    time = np.zeros(n_iter)
    for i in range(n_iter):
        seed = 100+i

        #Extract parameters from data
        y_test,N_test,K,D = load_data(type,seed)
        pi_1,mu_1,sigma_1,pi_2,mu_2,sigma_2,time[i]  = load_posterior(method,type,seed,K)
        #ll_test_1[i] = gll.lppd(y_test[0,].reshape(-1,D),pi_1,mu_1, sigma_1,K)
        #ll_test_2[i] = gll.lppd(y_test[1,].reshape(-1,D),pi_2,mu_2, sigma_2,K)
        ll_test_1[i] = gll.lppd(y_test[0,],pi_1,mu_1, sigma_1,K)
        ll_test_2[i] = gll.lppd(y_test[1,],pi_2,mu_2, sigma_2,K)
        

    print('For {}, dataset {}'.format(method,type))
    print(np.mean(ll_test_1/N_test))
    print(np.std(ll_test_1/N_test))
    print(np.mean(ll_test_2/N_test))
    print(np.std(ll_test_2/N_test))

    print(np.mean(time))
    print(np.std(time))

def main():
    eval('RRNPL','sep')
    eval('FINPL','sep')
    #eval('NUTS','sep')
    #eval('ADVI','sep')

if __name__=='__main__':
    main()