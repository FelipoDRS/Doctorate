# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:00:51 2020

@author: Felipo Soares
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_120820'

import os

files=os.listdir(path_image)

dados=[];nomes=[]
for i in files:
    if '.csv' in i:
        df_temp=pd.read_csv(path_image+'\\'+i,index_col=0)
        dados.append(df_temp.copy());nomes.append(i)
myorder = [4,0,2,1,3]
dados_re = [dados[i] for i in myorder]
        
bool_neg_1208=np.logical_and(dados[4]['Y pos']>590,np.logical_or(dados[4]['X pos']<300,np.logical_and(dados[4]['X pos']>540,dados[4]['X pos']<700)))

bool_neg_pl1_1308=np.logical_and(dados[0]['Y pos']>580,dados[0]['X pos']<300)

bool_neg_pl2_1308=dados[2]['Y pos']>380
bool_neg_pl2_1308.loc[36:37]=True
bool_neg_pl2_1308.loc[87:88]=False
 
bool_neg_pl1_1408=np.logical_and(dados[1]['Y pos']>580,dados[1]['X pos']<300)

bool_neg_pl2_1408=dados[3]['Y pos']>360
bool_neg_pl2_1408.loc[36:37]=True
bool_neg_pl2_1408.loc[87:88]=False

bool_neg=[bool_neg_pl1_1308,bool_neg_pl1_1408,bool_neg_pl2_1308,bool_neg_pl2_1408,bool_neg_1208]
bool_neg_re = [bool_neg[i] for i in myorder]

neg_vals=[]
for i in range(len(dados_re)):
    neg_val=dados_re[i].loc[bool_neg_re[i],:]
    neg_vals.append(neg_val.copy())

from sklearn.metrics import confusion_matrix
for i in range(len(dados_re)):
    negs=neg_vals[i]
    if neg_vals[i].shape[0]>5:
        corte=np.mean(negs)+2*np.std(negs,ddof=0)
    else:
        corte=np.max(negs)
    res_elisa=np.zeros(96)
    res_elisa[dados_re[i]['absorb']>corte['absorb']]=1
    res_elisa[dados_re[i]['absorb']>2*corte['absorb']]=2
    true_val=(1-bool_neg[i])*2
    print(confusion_matrix(true_val,res_elisa))
    print(np.mean(true_val==res_elisa))

for i in range(len(dados_re)):
    negs=neg_vals[i]
    if neg_vals[i].shape[0]>5:
        corte=np.mean(negs)+2*np.std(negs,ddof=0)
    else:
        corte=np.max(negs)
    res_elisa=np.zeros(96)
    res_elisa[dados_re[i]['sat median']>corte['sat median']]=1
    res_elisa[dados_re[i]['sat median']>2*corte['sat median']]=2
#    true_val=np.zeros(dados[i].shape[0])
#    true_val[dados[i]['sat median']>corte['sat median']]=1
#    true_val[dados[i]['sat median']>2*corte['sat median']]=2
    true_val=(1-bool_neg_re[i])*2

    print(confusion_matrix(true_val,res_elisa)[[0,2],:])
    cm=confusion_matrix(true_val,res_elisa)
    print([cm[0,0],cm[2,2]]/np.sum(cm,axis=1)[[0,2]])
    print(np.mean(true_val==res_elisa))
    
    
for i in range(len(dados_re)):
    negs=neg_vals[i]
    negs['sat2']=(negs['sat median']/128)**2
    dados_re[i]['sat2']=(dados_re[i]['sat median']/128)**2
    if neg_vals[i].shape[0]>5:
        corte=np.mean(negs)+2*np.std(negs,ddof=0)
    else:
        corte=np.max(negs)
    res_elisa=np.zeros(96)
    res_elisa[dados_re[i]['sat2']>corte['sat2']]=1
    res_elisa[dados_re[i]['sat2']>2*corte['sat2']]=2
    true_val=np.zeros(dados[i].shape[0])
    true_val[dados_re[i]['absorb']>corte['absorb']]=1
    true_val[dados_re[i]['absorb']>2*corte['absorb']]=2
    true_val=(1-bool_neg_re[i])*2
    cm=confusion_matrix(true_val,res_elisa)

#    print(cm[[0,2],:])
 #   print([cm[0,0],cm[2,2]]/np.sum(cm,axis=1)[[0,2]])
    print(cm)
    print(np.mean(true_val==res_elisa))

for i in range(len(dados_re)):
    negs=neg_vals[i]
    negs['sat2']=(negs['sat median']/128)**2
    dados_re[i]['sat2']=(dados_re[i]['sat median']/128)**2
    if neg_vals[i].shape[0]>5:
        corte=np.mean(negs)+2*np.std(negs,ddof=0)
    else:
        corte=np.max(negs)
    res_elisa=np.zeros(96)
    res_elisa[dados_re[i]['sat2']>corte['sat2']]=1
    res_elisa[dados_re[i]['sat2']>2*corte['sat2']]=2
    true_val=np.zeros(dados[i].shape[0])
    true_val[dados_re[i]['absorb']>corte['absorb']]=1
    true_val[dados_re[i]['absorb']>2*corte['absorb']]=2
#    true_val=(1-bool_neg_re[i])*2
    cm=confusion_matrix(true_val,res_elisa)

#    print(cm[[0,2],:])
 #   print([cm[0,0],cm[2,2]]/np.sum(cm,axis=1)[[0,2]])
    print(cm)
    print(np.mean(true_val==res_elisa))
    
    
    #%% bow the exponential
    
for i in range(len(dados_re)):
    negs=neg_vals[i]
    negs['sat exp']=np.exp(negs['sat median']/64)
    dados_re[i]['sat exp']=np.exp(dados_re[i]['sat median']/64)
    if neg_vals[i].shape[0]>5:
        corte=np.mean(negs)+2*np.std(negs,ddof=0)
    else:
        corte=np.max(negs)
    res_elisa=np.zeros(96)
    res_elisa[dados_re[i]['sat exp']>corte['sat exp']]=1
    res_elisa[dados_re[i]['sat exp']>2*corte['sat exp']]=2
    true_val=np.zeros(dados[i].shape[0])
    true_val[dados_re[i]['absorb']>corte['absorb']]=1
    true_val[dados_re[i]['absorb']>2*corte['absorb']]=2
#    true_val=(1-bool_neg_re[i])*2
    cm=confusion_matrix(true_val,res_elisa)

#    print(cm[[0,2],:])
#    print([cm[0,0],cm[2,2]]/np.sum(cm,axis=1)[[0,2]])
#    print(cm)
    print(np.mean(true_val==res_elisa))

    
    
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
for i in dados_re:
    X=i['sat median'].values.reshape(-1, 1);y=i['absorb']
    LR.fit(X,y)
    print(LR.intercept_,LR.coef_,LR.score(X,y))
    print(np.min(i)/np.mean(i))

for i in dados:
    X=(i['sat median'].values.reshape(-1, 1)/255)**2;y=i['absorb']
    LR.fit(X,y)
    print(LR.intercept_,LR.coef_,LR.score(X,y))
import statsmodels.api as sm

for i in dados:
    X=(i['sat median'].values.reshape(-1, 1)/96)**2;y=i['absorb']
    X=sm.add_constant(X)
    model = sm.OLS(y,X)
    res=model.fit()
#    print(res.summary())
#    print(res.params.values)
    print(res.mse_model,res.mse_resid)
    
    
plt.figure(); plt.plot(dados_re[2]['absorb'],dados_re[2]['sat median'],'+');
plt.xlabel('absorvancia');plt.ylabel('saturacao');plt.ylim([0,plt.ylim()[1]])
plt.title('placa 2 13/08')

plt.figure(); plt.plot(dados_re[2]['absorb'],dados_re[2]['sat median']**2,'+')
plt.xlabel('absorvancia');plt.ylabel('saturacao^2');plt.ylim([0,plt.ylim()[1]])
plt.title('placa 2 13/08')
