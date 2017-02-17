# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 18:28:56 2017

@author: Michael Williams
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import corner
import example_functions as ef
import adaptive_jumping_algorithms as aja

#from pymc3.stats import autocorr



time = np.arange(0,10.0,0.01)
data = ef.sinesignal(1.7,5.62,3.14,10.0,0.01,0.0,1.0, True)
signal = ef.sinesignal(1.7,5.62,1.0)

likelihood = ef.sinelikelihood
prior = ef.sineprior
"""
0 = MH
1 = AP   (computationally intesive)
2 = AM
"""
jumptype = 2
#parameters to estimate
parameters = [np.array([1.0, 5.0, 0.0])]    # A , f , phi
 #trace
trace = [likelihood(parameters[0], data)]
Ps = [np.array([0.0])]

# define stepsize for chain
stepsize = np.array([50.0, 50.0, 2*np.pi])
# count number of accepted values
accepted = 0.0
random_accepted = 0.0
# number of interations
N = 10000
#parameters for APjump
k = 0
K = 0
H = 300
U = 300
#parameters for AMjump
t0 = 100
E = np.array([0.01, 0.1, 0.01])
initial_stepsize = np.array([50.0, 50.0, 2*np.pi])
#record all proposed jumps
jumps = [0]


# metropolis-hastings
for i in range(N):
    # old parameter values
    old_parameter = parameters[len(parameters)-1]
    # old numerator of parameters for parameters
    old_like = likelihood(old_parameter,data)
    old_prior = prior(old_parameter,data)
    
    #choose jump type
    if jumptype == 0 :
        #new parameter using Metropolis Hastings
        new_parameter = aja.MHjump(parameters, stepsize)
    if jumptype == 1 :
        # new parameter using AM jump
        K,new_parameter = aja.APjump(parameters, initial_stepsize,K, H, U, i)
    if jumptype == 2:
        #new parameter using AP jump
        new_parameter = aja.AMjump(parameters, initial_stepsize, t0, E, i)
            
    # make it a vector
    new_parameter = new_parameter[0,:]
    # evaluate the parameter
    new_like = likelihood(new_parameter,data)
    new_prior = prior(new_parameter, data)
        
    # accept or reject
    P = (new_like + new_prior) - (old_like + old_prior)
    Ps.append(P)
    
    if (P > 0): # 0 for loglikelihood 1 for normal
        parameters.append(new_parameter)
        accepted += 1.0
        
    else:
        j = (np.random.rand())
        lnj = np.log(j)
        if (i > H) and (lnj < P) :
            parameters.append(new_parameter)
            accepted += 1.0
            random_accepted +=1.0
        else:
            parameters.append(old_parameter)
   
#burn in 50%                         
result = parameters#[N/2:]
estimated_parameters = result[-1]
A = estimated_parameters[0]
f = estimated_parameters[1]
phi = estimated_parameters[2]
estimated_signal = ef.sinesignal(A,f,phi)
percent = accepted/N
rand_percent = random_accepted/N
print ("accepted interations = '{0}', random accepted iterations = '{1}'" .format(percent, rand_percent))
print ("A = '{0}', f = '{1}', phi = '{2}'" .format(A, f, phi))

X = np.zeros(len(result)) # A
Y = np.zeros(len(result)) # f
Z = np.zeros(len(result)) # phi
W = np.zeros(len(result))
for l in range(len(result)-1):
    X[l] = result[l][0]
    Y[l] = result[l][1]
    Z[l] = result[l][2]
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
#plots for signals
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(time, data, 'b')
ax1.set_title('Comparison of two signals')
ax2.plot(time, signal, 'b')
ax3.plot(time, estimated_signal, 'r')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
"""
plt.plot(time, signal, 'b')
plt.plot(time, estimated_signal, 'r')
"""

"""
#plt.plot(its,V,'.y')
plt.figure()
plt.plot(its, trace)
plt.figure()
plt.plot(its, Ps, 'x')

#figure = corner.corner(corner_data)
"""