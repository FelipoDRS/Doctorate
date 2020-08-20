# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:54:52 2020

@author: Felipo Soares
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_250620'

files= os.listdir(path_image)
DF_dados=[]
names=[]
for i in files:
    if '.csv' in i:    
        DF_temp = pd.read_csv(path_image+'\\'+i, index_col=0)
        DF_temp['X pos']=np.round((DF_temp['X pos']-DF_temp['X pos'].min())/(DF_temp['X pos'].max()-DF_temp['X pos'].min())*11)
        DF_temp['Y pos']=np.round((DF_temp['Y pos']-DF_temp['Y pos'].min())/(DF_temp['Y pos'].max()-DF_temp['Y pos'].min())*7)
        DF_temp['exp sat']=np.exp(DF_temp['sat']/64)
        DF_dados.append(DF_temp)
        names.append(i)


negative=[0,3,6,9]
positive=[1,2,4,5,7,8,10,11]
dil1=[0,4];dil2=[1,5];dil3=[2,6];dil4=[3,7];
dil=[dil1,dil2,dil3,dil4]
#dil=[0,1,2,3,4,5,6,7]
import time
for k in DF_dados:
#    time.sleep(10)
    dados=k.values
    for j in dil:
        dados_neg=dados[np.logical_and(np.in1d(dados[:,0],negative),np.in1d(dados[:,1],j)),:]
        dados_pos=dados[np.logical_and(np.in1d(dados[:,0],positive),np.in1d(dados[:,1],j)),:]
        mean_neg=dados_neg.mean(axis=0)
        std_neg=dados_neg.std(axis=0)
        mean_pos=dados_pos.mean(axis=0)
        acerto=mean_neg+3*std_neg<dados_pos
#        print(np.mean(acerto,axis=0),j)
        print(dados_pos.max(axis=0)[4]-dados_neg.min(axis=0)[4])
        time.sleep(1.2)


for k in DF_dados:
#    time.sleep(10)
    dados=k.values
    for j in dil:
        dados_neg=dados[np.logical_and(np.in1d(dados[:,0],negative),np.in1d(dados[:,1],j)),:]
        dados_pos=dados[np.logical_and(np.in1d(dados[:,0],positive),np.in1d(dados[:,1],j)),:]
        mean_neg=dados_neg.mean(axis=0)
        std_neg=dados_neg.std(axis=0)
        mean_pos=dados_pos.mean(axis=0)
        std_pos=dados_pos.std(axis=0)
        
        acerto=dados_neg<mean_pos-3*std_neg
        print(np.mean(acerto,axis=0),j)
        time.sleep(1)


LR=LinearRegression()
X_train=dados_train[:,2]
Y_train=dados_train[:,4]

X_test=dados_test[:,2]
Y_test=dados_test[:,4]

LR.fit(X_train.reshape((-1,1)),Y_train)
print(LR.score(X_test.reshape((-1,1)),Y_test))

plt.plot(X_train,Y_train,'b+')
plt.plot(X_test,Y_test,'r+')

negative=[0,3,6,9]
positive=[1,2,4,5,7,8,10,11]
dil1=[0,4];dil2=[1,5];dil3=[2,6];dil4=[3,7];
dados_neg_train=dados_train[np.in1d(dados_train[:,0],negative),:]
dados_pos_train=dados_train[np.in1d(dados_train[:,0],positive),:]

print(dados_neg_train.mean(axis=0))
print(dados_neg_train.std(axis=0))
print(dados_pos_train.mean(axis=0))
print(dados_pos_train.std(axis=0))

mean_neg=dados_neg_train.mean(axis=0)
std_neg=dados_neg_train.std(axis=0)
mean_pos=dados_pos_train.mean(axis=0)
std_pos=dados_pos_train.std(axis=0)

errors=mean_neg+3*std_neg<dados_pos_train
print(np.mean(mean_neg+3*std_neg<dados_pos_train,axis=0))

dados_neg_dil1=dados_train[np.logical_and(np.in1d(dados_train[:,0],negative),np.in1d(dados_train[:,1],dil1)),:]
dados_pos_dil1=dados_train[np.logical_and(np.in1d(dados_train[:,0],positive),np.in1d(dados_train[:,1],dil1)),:]

print(dados_neg_dil1.mean(axis=0))
print(dados_neg_dil1.std(axis=0))
print(dados_pos_dil1.mean(axis=0))
print(dados_pos_dil1.std(axis=0))

mean_neg=dados_neg_dil1.mean(axis=0)
std_neg=dados_neg_dil1.std(axis=0)
mean_pos=dados_pos_dil1.mean(axis=0)
std_pos=dados_pos_dil1.std(axis=0)

errors=mean_neg+3*std_neg<dados_pos_dil1
print(np.mean(errors,axis=0))

plt.figure();plt.plot(dados_neg_dil1[:,2],dados_neg_dil1[:,3],'b+')
plt.plot(dados_pos_dil1[:,2],dados_pos_dil1[:,3],'r+')