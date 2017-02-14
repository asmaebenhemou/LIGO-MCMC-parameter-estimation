# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 18:28:56 2017

@author: Zxilix
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import corner
import example_functions as ef

from pymc3.stats import autocorr
# start by defining likelihood function - lf

"""
functions to try:
    
circle:
    x = alpha[0]
    y = alpha[1]
    r2 = x**2+y**2
    if (r2 <= 1):
        r2 = r2
    else:
        r2 = 0
    return r2 
    
two gaussians:
    x = alpha[0]
    y = alpha[1]
    g1 = mlab.bivariate_normal(x, y, 1.0, 1.0, -1, -1, -0.8)
    g2 = mlab.bivariate_normal(x, y, 1.5, 0.8, 1, 2, 0.6)
    return 0.6*g1+28.4*g2/(0.6+28.4)
    

    
"""
data = ef.sinesignal(1.7,5.62,1.0,10.0,0.01,0.0,1.0, True)

likelihood = ef.sinelikelihood
prior = ef.sineprior

#parameters to estimate
parameters = [np.array([1.0, 5.0, 1.0])]    # A , f , phi
 #trace
trace = [likelihood(parameters[0], data)]
Ps = [np.array([0.0])]

# define stepsize for chain
stepsize = np.array([5.0, 25.0, 5.0])
# count number of accepted values
accepted = 0.0
random_accepted = 0.0
# number of interations
N = 10000

# metropolis-hastings
for i in range(N):
    # old parameter values
    old_parameter = parameters[len(parameters)-1]
    # old numerator of parameters for parameters
    old_like = likelihood(old_parameter,data)
    old_prior = prior(old_parameter,data)
    
    # new parameters to try:
    #covariance for the normal dist
    covar = stepsize * np.diag(np.ones(len(old_parameter)))
    #proposed jump
    jump = np.random.multivariate_normal([0,0,0], covar, 1)
    #new parameter
    new_parameter = old_parameter + jump
    # make it a vector
    new_parameter = new_parameter[0,:]
    # evaluate the parameter
    new_like = likelihood(new_parameter,data)
    new_prior = prior(new_parameter, data)
        
    # accept or reject
    P = (new_like + new_prior) - (old_like + old_prior)
    Ps.append(P)
    if (P > 1):
        parameters.append(new_parameter)
        accepted += 1.0
        
    else:
        j = (np.random.rand())
        lnj = np.log(j)
        if (lnj < P) :
            parameters.append(new_parameter)
            accepted += 1.0
            random_accepted +=1.0
        else:
            parameters.append(old_parameter)
                           
result = parameters#[N/10:]   
percent = accepted/N
rand_percent = random_accepted/N
print ('accepted interations = %.5f, random accepted iterations = %.5f' % (percent, rand_percent))

X = np.zeros(len(result)) # A
Y = np.zeros(len(result)) # f
Z = np.zeros(len(result)) # phi
W = np.zeros(len(result))
for k in range(len(result)-1):
    X[k] = result[k][0]
    Y[k] = result[k][1]
    Z[k] = result[k][2]
    #W[k] = result[k][3]


its = np.array(range(len(result)))

#Xdata = np.column_stack((its, X))
#corner_data = np.vstack(Xdata)



plt.figure()
plt.plot(its, X, '.g')
plt.figure()
plt.plot(its,Y,'.r')
plt.figure()
plt.plot(its,Z,'.b')
plt.figure()
"""
#plt.plot(its,V,'.y')
plt.figure()
plt.plot(its, trace)
plt.figure()
plt.plot(its, Ps, 'x')

#figure = corner.corner(corner_data)
"""