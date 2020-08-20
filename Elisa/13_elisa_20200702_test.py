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

path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_20200702'

files= os.listdir(path_image)
DF_dados=[]
names=[]
for i in files:
    if '.csv' in i and 'b_placa' in i:    
        DF_temp = pd.read_csv(path_image+'\\'+i, index_col=0)
        DF_temp['X pos']=np.round((DF_temp['X pos']-DF_temp['X pos'].min())/(DF_temp['X pos'].max()-DF_temp['X pos'].min())*11)
        DF_temp['Y pos']=np.round((DF_temp['Y pos']-DF_temp['Y pos'].min())/(DF_temp['Y pos'].max()-DF_temp['Y pos'].min())*7)
        DF_temp['exp sat']=np.exp(DF_temp['sat IQR']/64)
        DF_temp['sat^2']=DF_temp['sat IQR']*DF_temp['sat IQR']
        
        DF_dados.append(DF_temp)
        names.append(i)

titul_collect=[];
branco_collect=[];
analise_collect=[];
for i in DF_dados:
    titul_bool=i['X pos']==0
    branco_bool=np.logical_and(np.logical_and(np.logical_and(i['X pos']>=1,i['X pos']<=2),i['Y pos']>=0),i['Y pos']<=3)
    analise_bool=np.logical_not(np.logical_or(branco_bool,titul_bool))
    titulacao=i.loc[titul_bool,:].copy();branco=i.loc[branco_bool,:].copy();analise=i.loc[analise_bool,:].copy()
    titul_collect.append(titulacao);branco_collect.append(branco);analise_collect.append(analise)
    
#import time
for i in range(len(titul_collect)):
    mean_neg=np.mean(branco_collect[i],axis=0)
    std_neg=np.std(branco_collect[i],axis=0,ddof=1)
    true_neg=analise_collect[i]['absorb'].values<mean_neg['absorb']+std_neg['absorb']*3
    test0_neg=analise_collect[i]['sat median'].values<mean_neg['sat median']+std_neg['sat median']*3
    test1_neg=analise_collect[i]['exp sat'].values<mean_neg['exp sat']+std_neg['exp sat']*3
    test2_neg=analise_collect[i]['sat^2'].values<mean_neg['sat^2']+std_neg['sat^2']*8
    
    print(mean_neg['absorb']+std_neg['absorb']*3,mean_neg['sat median']+std_neg['sat median']*3)
#    print(analise_collect[i].loc[true_neg!=test0_neg].iloc[:,2:])
    print(np.sum(true_neg==test0_neg))
    print(np.sum(true_neg==test1_neg))
    print(np.sum(true_neg==test2_neg))
    

for i in range(len(titul_collect)):
    mean_neg=np.mean(branco_collect[i].values,axis=0)
    std_neg=np.std(branco_collect[i].values,axis=0)
    close=np.logical_and(analise_collect[i]['absorb'].values<(mean_neg[5]+std_neg[5]*3)*1.1,
                         analise_collect[i]['absorb'].values>(mean_neg[5]+std_neg[5]*3)*0.9)
    print(np.sum(close))
    
for i in range(len(titul_collect)):
    titul_collect[i].plot(x='sat mean',y='absorb',style='o')    
for i in range(len(titul_collect)):
    analise_collect[i].plot(x='sat mean',y='absorb',style='o')    

analise_collect[0].plot(x='sat median',y='absorb',style='+',markersize=10)
plt.plot([-5,175],[0.1069,0.1069],'r--',label='limite de detecção ELISA')
plt.plot([31,31],[0.00,0.8],'g--',label='limite de detecção ELISA')
plt.xlim([-5,60]);plt.ylim([0,0.2])
plt.title('Placa 1');plt.ylabel('Absorvancia')
plt.text(42,0.04,'III');plt.text(42,0.17,'II')
plt.text(13,0.17,'I');plt.text(13,0.04,'IV')
plt.savefig(path_image+'\\detec_placa1.jpg', dpi=200)

analise_collect[2].plot(x='sat median',y='absorb',style='+',markersize=10)
plt.plot([-5,175],[0.08903,0.08903],'r--')
plt.plot([21.28,21.28],[0.00,0.8],'g--')
plt.xlim([-5,50]);plt.ylim([0,0.18])
plt.title('Placa 2');plt.ylabel('Absorvancia')
plt.text(32,0.030,'III',fontsize=15);plt.text(32,0.14,'II',fontsize=15)
plt.text(11,0.14,'I',fontsize=15);plt.text(11,0.030,'IV',fontsize=15)
plt.savefig(path_image+'\\detec_placa2.jpg', dpi=200)

analise_collect[2].plot(x='sat median',y=['sat IQR','sat mean'],style='+')
plt.plot([0,175],[0.08903,0.08903],'r--')
plt.plot([21.28,21.28],[0.00,0.8],'g--')
plt.xlim([0,50]);plt.ylim([0,50])
