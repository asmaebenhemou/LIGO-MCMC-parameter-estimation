from __future__ import division
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import seaborn as sns

# need data from file X
def estimated_autocorrelation(data):
    
    n = len(data)
    variance = data.var()
    data = data - data.mean()
    r = np.correlate(data, data, mode='full')[-n:]
    #assert np.allclose(r, np.array([data[:n-k]*data[-(n-k):]).sum() for k in range(n)])) # check calculation
    autocorrelation = r/(variance*(np.arange(n, 0, -1)))
   
    return autocorrelation
#----------------------------------------------------------------------------------------------------------
#test with data.txt file:
#data = loadtxt(X,unpack=True,usecols=[1])
#time = loadtxt(data,unpack=True,usecols=[0])
#plt.plot(time, estimated_autocorrelation(data))
#plt.xlabel('time in second')
#plt.ylabel('Estimated autocorrelation')
#plt.savefig('C:\\Users\sony\Documents\LIGO gravitational waves honours lab\Monte Carlo methods\autocorrelation.png')

#----------------------------------------------------------------------------------------------------------
# Voir seconde maniere de faire:
#x = series - np.mean(data)  # get the auto-correlation 
#z = np.conv(x, x, mode='full')
#z = np.fft.ifftshift(z, axes=None)
#autocorrelation = z[1:len(series)]
#autocorrelation = autocorrelation / autocorrelation[1] 