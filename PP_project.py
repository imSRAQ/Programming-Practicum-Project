# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:39:04 2022

@author: hp
"""
import numpy as np
from random import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def _sigmoid(x):

    y = 1.0 / (1 + np.exp(-x))
    return y

def _mse(target, output):
    return np.average((target - output) ** 2)


try:
    df=pd.read_csv("C:\\Users\hp\Downloads\Sonar221.csv")
    y1=df.loc[:,'label']
    y2=[]
    for i in y1:
        l=[]
        l.append(i)
        y2.append(l)
    yl=np.array(y2) 
    df.drop('label',inplace=True, axis=1)  
    xp=np.array(df) 
    x, x_test, y, y_test = train_test_split(xp, yl, train_size=0.8)
    m,n=x.shape
    network=[n,5,1]

    weights = []
    bias=[]
    for i in range(len(network) - 1):
        w = np.random.rand(network[i], network[i + 1])
        b= np.random.rand(network[i+1],1)
        weights.append(w)
        bias.append(b)

    epochs=50
    lr=0.1
    e=list(np.arange(epochs))
    L=[]
    for i in range(epochs):
        sum_errors = 0

         
        for j, input in enumerate(x):
            inp=[]
            for k in input:
                l=[]
                l.append(k)
                inp.append(l)
            input=np.array(inp)    
                
            y1 = y[j]
            z1=np.matmul(np.transpose(weights[0]),input)+bias[0]
            a1= _sigmoid(z1)
            one=[]
            for m in range(len(a1)):
                o=[]
                o.append(1)
                one.append(o)
            one=np.array(one)    
                
            z2=np.matmul(np.transpose(weights[1]),a1)+bias[1]
            a2= _sigmoid(z2)
            error2= y1-a2
            der_a2=np.matmul(a2,np.transpose(np.ones(len(a2))-a2))
            der_L_w2=np.matmul(np.transpose(error2),np.matmul(der_a2,a1.T))
            der_L_b2=np.matmul(error2,der_a2)
            error1=np.matmul(error2.T,np.matmul(der_a2,weights[1].T))
            der_a1=np.matmul(a1,(one-a1).T)
            der_L_w1=np.matmul(error1.T,np.matmul(a1.T,np.matmul((one-a1),input.T)))
            der_L_b1=np.matmul(error1,der_a1)
            
            weights[1]=weights[1]+lr*(der_L_w2.T)
            
            weights[0]=weights[0]+lr*(der_L_w1.T)
            
            
            bias[1]=bias[1]+lr*(der_L_b2.T)
            bias[0]=bias[0]+lr*(der_L_b1.T)
            
            sum_errors= sum_errors+_mse(y1,a2)
            
        print("Error: {} at epoch {}".format(sum_errors / len(x), i+1))
        L.append(sum_errors / len(x))
        
    plt.plot(e,L)
    plt.show()    
        

    #Testing our neural network
    pred=[]
    for j, input in enumerate(x_test):
        inp=[]
        for k in input:
            l=[]
            l.append(k)
            inp.append(l)
        input=np.array(inp)    
            
        y1 = y_test[j]
        z1=np.matmul(np.transpose(weights[0]),input)+bias[0]
        a1= _sigmoid(z1)
        z2=np.matmul(np.transpose(weights[1]),a1)+bias[1]
        a2= _sigmoid(z2)
        pred.append(a2)


    cal=[]    
    for i in range(len(y_test)):
        if pred[i][0]>=0.5:
            cal.append(1)
        else:
            cal.append(0)
            
        
    from sklearn.metrics import accuracy_score
    A=accuracy_score(y_test, cal)
    print(A)

except FileNotFoundError:
    print("File not found.")
    
        
        
    


        
    


    
    