# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:36:11 2021

@author: 61415
"""
from numpy import mean
from numpy import std
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from numpy import mean
from numpy import std
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import warnings
import scipy.stats as stats
import scipy.linalg
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random
with open('random30_4.pickle', 'rb') as f:
    groupsAll = pickle.load(f)
modelCh = 'svr'
warnings.filterwarnings(action='ignore')

t11 = [12.8400,   11.7200,   10.6000 ,   9.4000  ,  8.3200   , 7.2400]
t1104 = [  13.7600 ,  12.3200 ,  11.0000 ,   9.9200,   8.6800,    7.8000]
t1108 = [14.4400 ,  12.9200 ,  11.6800  , 10.6000 ,   9.6800 ,   8.7600]
t1112= [15.0000 ,  13.7200  , 12.4800 ,  11.4000 ,  10.3600  ,  9.3600]
t1116 = [15.9200  , 14.4800 ,  13.3200  , 12.0800  , 11.1600 ,  10.1600]
t1120 = [16.2400  , 15.1200  , 14.0800 ,  13.1200  , 11.8400  , 10.9200]

valt = 0

for group in range(len(groupsAll)):
    #tref_1112 = -0.276 * hum + 25.887
    
    bplotH = {'40':[],'44':[],'48':[],'52':[],'56':[],'60':[]}
    bplotT = {'22.67':[], '22.58':[],'22.5':[],'22.41':[],'22.33':[],'22.24':[]}
    
    # notch  6= 12.84, 12.92 	12.4800 11.40 11.1600 10.920
    notchFreqOriginal = [12.84	,13.76	,14.44,	15,	15.92,	16.24,
    11.72,	12.32,	12.92	,13.72,	14.48,	15.12,
    10.60,	11,	11.68,	12.48,	13.32,	14.08,
    9.400,	9.920,	10.60,	11.40	,12.08	,13.12,
    8.320,	8.680,	9.680	,10.36	,11.16	,11.84,
    7.240,	7.800,	8.760,	9.360	,10.16,	10.92]
    
    dcArrOriginal = [1.97, 2.01, 1.86, 1.50, 1.56, 1.15,
             2.03,1.90,1.63,1.44,1.33,0.97,
             1.93, 1.62, 1.54, 1.53, 1.41, 1.11,
             1.83, 1.64, 1.55, 1.37, 1.20, 1.09,
             1.74, 1.55, 1.42, 1.29, 1.14, 1.03,
             1.84,1.59,1.49,1.30,0.99,0.98]
    
    RHOriginal = [39.80	,44	,48	,52,	55.60,	59.70,
    40,	44.20,	48,	52.10,	56.20,	60.50,
    40,	44.30,	48.30,	52.30,	56.30,	60.50,
    39.90,	44.10,	48.20,	52,	56,	60.50,
    39.90,	44.40,	48.10,	52.20,	56.20,	60,
    40.40,	44.10,	47.60,	51.70,	56	,59.80]
    
    training = {'notchFreq':[], 'dcArr':[], 'RH':[]}
    testing = {'notchFreq':[], 'dcArr':[], 'RH':[]}
    
    for i in range(len(notchFreqOriginal)):
        if  i%6 != i:
            training['notchFreq'].append(notchFreqOriginal[i])
            training['dcArr'].append(dcArrOriginal[i])
            training['RH'].append(RHOriginal[i])
        else:
            testing['notchFreq'].append(notchFreqOriginal[i])
            testing['dcArr'].append(dcArrOriginal[i])
            testing['RH'].append(RHOriginal[i])
            
    
    y = []
    X = []
    hum = [40,44,48,52,56,60]
    temp = [0,4,8,12,16,20]
    tempC = [22.67,22.58,22.5,22.41,22.33,22.24]
    
    
    for i in range(len(training['RH'])):
        # y value ground truth
        RH_val = training['RH'][i]
        frequency_equivalent = -0.276 * RH_val + 25.887
        y.append(frequency_equivalent)
        
        
    
    for i in range(len(training['RH'])):
        f1s = ( training['notchFreq'][i] - (min(notchFreqOriginal)))/max(notchFreqOriginal)
        f3s = ( training['dcArr'][i] - min(dcArrOriginal))/max(dcArrOriginal)
        f1n = ( training['notchFreq'][i] - (mean(notchFreqOriginal)))/std(notchFreqOriginal)
        f3n = ( training['dcArr'][i] - mean(dcArrOriginal))/std(dcArrOriginal)
    
        X.append([f1s,f3s])
    
    regressor = SVR(kernel = 'rbf',C =50, gamma=2.4, epsilon =  0.1 )
    
    #regressor = MLPRegressor(hidden_layer_sizes = 17, activation='relu',solver='lbfgs',alpha=0.00001, learning_rate='adaptive')
    model = MultiOutputRegressor(regressor)
    regressor.fit(X, y)
    
    sumerr = 0
    rh_array = []
    
    temp1 = []
    temp2 = []
    temp3 = []
    # now we do testing
    for j in range(len(testing['RH'])):
        f1s = ( testing['notchFreq'][j] - (min(notchFreqOriginal)))/max(notchFreqOriginal)
        f3s = ( testing['dcArr'][j] - min(dcArrOriginal))/max(dcArrOriginal)
        f1n = ( testing['notchFreq'][j] - (mean(notchFreqOriginal)))/std(notchFreqOriginal)
        f3n = ( testing['dcArr'][j] - mean(dcArrOriginal))/std(dcArrOriginal)
        frequency_prediction = regressor.predict([[f1s, f3s]])
        RH_prediction_equivalent = (frequency_prediction - 25.887)/-0.276
        
        RH_val = testing['RH'][j]
        frequency_truth = -0.276 * RH_val + 25.887
        
        #print(RH_val)
        #print(RH_prediction_equivalent[0])
        #print((RH_prediction_equivalent[0] -RH_val)**2 )
        #print(frequency_prediction[0])
        #print(frequency_truth)    
        #print("-----")
        
        temp1.append(RH_val)
        temp2.append(RH_prediction_equivalent[0])
        temp3.append(abs(RH_prediction_equivalent - RH_val)[0])
        sumerr += abs(RH_prediction_equivalent - RH_val)
    
    #print(temp1)
    #print(temp2)
    #print(temp3)
        
    #print(sumerr/6)
    valt += sumerr/6

print(valt/30)
    
    
