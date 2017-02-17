# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:23:22 2017
Adaptive Jumping Algorithms
@author: Michael Williams
"""

import numpy as np
#from scipy import stats

"""
Standard Metropolis Hastings

"""
def MHjump(parameters, stepsize):
    par = np.array(parameters)
    _,d = par.shape
    covar = stepsize * np.identity(d)
    mean = np.zeros((d,))
    jump = np.random.multivariate_normal(mean, covar, 1)
    new_parameter = parameters[-1] + jump
    return new_parameter
    


"""
Adaptive proposal distribution for random walk Metropolis algorithm
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.3205&rep=rep1&type=pdf
"""
def APjump(parameters, initial_stepsize,K, H, U, i):
    # H - iterations to use as history
    # make sure parameters are an array not list
    par = np.array(parameters)
    #if at the beginning of the run, must find first states for i < H
    if i < H:
        h = H/2
        if i < h:
            #k = par
            _,d = par.shape
            covar = initial_stepsize * np.identity(d)
            mean = np.zeros((d,))
            #proposed jump
            new_parameter = par[-1] + np.random.multivariate_normal(mean, covar, 1)
        else:
            if i % (U/2) == 0:
                k = par[-h:]
                # centering matrix
                C = np.identity(h) - (1/h) * np.ones((h,h))
                # centered k
                K = np.dot(C,k)
                # covariance
                #R_t = 1/(i-1) * np.transpose(K) * K
            _,d = K.shape
            # scaling factor
            cd = 2.4 / np.sqrt(d)
            #mean and covariance for gaussian
            mean = np.zeros((h,))
            covar = np.identity(h)
            #random parameters from normlaized gaussian
            gauss = np.random.multivariate_normal(mean, covar, 1)
            #proposed jump
            new_parameter = par[-1] + (cd / np.sqrt(h-1)) * np.dot(K.T,gauss.T).T
        
    else:
        # check update frequecy
        if i % U == 0:
            # define H x d matrix k
            k = par[-H:]
            # centering matrix
            C = np.identity(H) - (1/H) * np.ones((H,H))
            # centered k
            K = np.dot(C,k)
        # covariance
        #R_t = 1/(H-1) * np.transpose(K) * K
        _,d = K.shape
        # scaling factor
        cd = 2.4 / np.sqrt(d)
        #mean and covariance for gaussian
        mean = np.zeros((H,))
        covar = np.identity(H)
        #random parameters from normlaized gaussian
        gauss = np.random.multivariate_normal(mean, covar, 1)
        #proposed jump - 
        new_parameter = par[-1] + (cd / np.sqrt(H-1)) * np.dot(K.T,gauss.T).T
    return K,new_parameter 

"""
An adaptive Metropolis algorithm
https://projecteuclid.org/download/pdf_1/euclid.bj/1080222083

requiers: positive definite initial covariance
          E > 0 
"""
def AMjump(parameters,initial_stepsize,t0,E,i):
    #t0 - start point for history
    # E - constant > 0 , very small compared to size of set in R^d
    # i - iteration
    # initial_stepsize - for intial covariance estimate
    # make sure parameters are an array not list
    par = np.array(parameters)
    _,d = par.shape
    # need initial values to base later predictions off
    if i <= t0:
        #covariance for first iterations when t<=t0
        Ct = initial_stepsize * np.identity(d)
        
    else:
        # dimension dependant parameter, this case is for gaussian
        sd = 2.4 ** 2 / d
        # covaraince of parameters so far
        covar = (np.cov(par[:,0]),np.cov(par[:,1]),np.cov(par[:,2])) * np.identity(d)
        #covariance matrix for gaussian
        Ct = sd * covar + sd * E * np.identity(d)
    mean = par[-1]   
    new_parameter = np.random.multivariate_normal(mean, Ct, 1) 
        
    return new_parameter
    
    

