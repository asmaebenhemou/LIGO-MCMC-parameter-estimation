# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:00:40 2017

@author: User
"""
import numpy as np

# Sinusoidal singal with gaussian noise
def sinesignal(amplitude, frequency, phase, duration=10.0, interval=0.01, mean=0.0, sigma=1.0, noise=False):
    time = np.arange(0, duration, interval)
    signal = amplitude * np.sin((2.0 * np.pi * frequency * time) + phase)
    if noise is True:
        n = np.random.normal(mean, sigma, size=len(time))
        final_signal = n + signal
    else:
        final_signal = signal
        
    return final_signal
        
# Sinusoidal signal likelihood
def sinelikelihood(parameters, data, duration=10.0, interval=0.01, sigma=1.0):
    amplitude, frequency, phase, = parameters
    signal = sinesignal(amplitude, frequency, phase, duration, interval)
    likelihood = (-0.5 *np.sum(((signal - data)**2.0 )/(sigma**2.0)))
    return likelihood

# Sinusoidal signal prior
def sineprior(parameters, data):
    amplitude, frequency, phase, = parameters
    prior = 0.0
    if (frequency < 0.0):
        prior = -10.0 ** 10.0
    return prior