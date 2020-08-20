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

def evaluate_sample(data,cut):
    result=np.ones((data.shape[0]));
    result[data/cut<=1]=0
    result[data/cut>=2]=2
    return result
    
        

files= os.listdir(path_image)
DF_dados=[]
names=[]
for i in files:
    if '.csv' in i and 'b_placa' in i:    
        DF_temp = pd.read_csv(path_image+'\\'+i, index_col=0)
        DF_temp['X pos']=np.round((DF_temp['X pos']-DF_temp['X pos'].min())/(DF_temp['X pos'].max()-DF_temp['X pos'].min())*11)
        DF_temp['Y pos']=np.round((DF_temp['Y pos']-DF_temp['Y pos'].min())/(DF_temp['Y pos'].max()-DF_temp['Y pos'].min())*7)
        DF_temp['exp sat']=np.exp(DF_temp['sat median']/64)
        DF_temp['sat^2']=DF_temp['sat median']*DF_temp['sat median']
        
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
from sklearn.metrics import confusion_matrix

for i in range(len(titul_collect)):
    mean_neg=np.mean(branco_collect[i],axis=0)
    std_neg=np.std(branco_collect[i],axis=0,ddof=1)
    true_cut=mean_neg['absorb']+2*std_neg['absorb']
    true_neg=evaluate_sample(analise_collect[i]['absorb'],true_cut)
    test0_cut=mean_neg['sat median']+std_neg['sat median']*2
    test0_neg=evaluate_sample(analise_collect[i]['sat median'],test0_cut)
    test1_cut=mean_neg['exp sat']+std_neg['exp sat']*2
    test1_neg=evaluate_sample(analise_collect[i]['exp sat'],test1_cut)
    test2_cut=mean_neg['sat^2']+std_neg['sat^2']*2
    test2_neg=evaluate_sample(analise_collect[i]['sat^2'],test2_cut)
#    test1_neg=analise_collect[i]['exp sat'].values<mean_neg['exp sat']+std_neg['exp sat']*3
 #   test2_neg=analise_collect[i]['sat^2'].values<mean_neg['sat^2']+std_neg['sat^2']*8
    
    print(np.unique(true_neg,return_counts=True))
#    print(analise_collect[i].loc[true_neg!=test0_neg].iloc[:,2:])
    print(confusion_matrix(true_neg,test0_neg))
    print(confusion_matrix(true_neg,test1_neg))
    print(confusion_matrix(true_neg,test2_neg))
  #  print(np.sum(true_neg==test1_neg))
   # print(np.sum(true_neg==test2_neg))
    

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


mean_placa1=np.mean(branco_collect[0],axis=0)
std_placa1=np.std(branco_collect[0],axis=0,ddof=1)

placa_1_true_cut=mean_placa1['absorb']+2*std_placa1['absorb']

mean_placa1=np.mean(branco_collect[0],axis=0)
std_placa1=np.std(branco_collect[0],axis=0,ddof=1)

test0_cut_placa1=mean_placa1['sat median']+2*std_placa1['sat median']

analise_collect[0].plot(x='sat median',y='absorb',style='+',markersize=10)
plt.plot([-5,175],[placa_1_true_cut,placa_1_true_cut],'r--',label='limite de detecção ELISA')
plt.plot([test0_cut_placa1,test0_cut_placa1],[0.00,0.8],'g--',label='limite de detecção ELISA')
plt.plot([-5,175],[placa_1_true_cut*2,placa_1_true_cut*2],'r--',label='limite de detecção ELISA')
plt.plot([test0_cut_placa1*2,test0_cut_placa1*2],[0.00,0.8],'g--',label='limite de detecção ELISA')
plt.xlim([-5,75]);plt.ylim([0,0.30])
plt.title('Placa 1');plt.ylabel('Absorvancia')
#plt.text(42,0.04,'III');plt.text(42,0.17,'II')
#plt.text(13,0.17,'I');plt.text(13,0.04,'IV')
plt.savefig(path_image+'\\detec_placa1_novo.jpg', dpi=200)

analise_collect[2].plot(x='sat median',y='absorb',style='+',markersize=10)
plt.plot([-5,175],[true_cut,true_cut],'r--')
plt.plot([test0_cut,test0_cut],[0.00,0.8],'g--')
plt.plot([-5,175],[true_cut*2,true_cut*2],'r--')
plt.plot([test0_cut*2,test0_cut*2],[0.00,0.8],'g--')
plt.xlim([-5,60]);plt.ylim([0,0.28])
plt.title('Placa 2');plt.ylabel('Absorvancia')
plt.savefig(path_image+'\\detec_placa2_novo.jpg', dpi=200)

analise_collect[2].plot(x='sat median',y=['sat IQR','sat mean'],style='+')
plt.plot([0,175],[0.08903,0.08903],'r--')
plt.plot([21.28,21.28],[0.00,0.8],'g--')
plt.xlim([0,50]);plt.ylim([0,50])
