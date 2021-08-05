# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:58:54 2021

@author: Felipo Soares
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate, ShuffleSplit, KFold
from sklearn.linear_model import LinearRegression, RidgeCV,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao.csv')
dados_dp=dados_dp.iloc[:,1:]
x_var_da=['T', 'RSO','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_C', 'd_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]
y_var_dp=['IV_R', 'RendR', 'RendDP', 'IV_DP']

dict_feature={'IV_R':['IV_C', 'd_C', 'IR_C', 'RSO_desaro', 'Lav_despar'],
               'd_R':['IV_C', 'd_C', 'IR_C', 'RSO_desaro', 'RSO_despar', 'Lav_despar'], 
               'IR_R':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'RendR':[ 'IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', 'Lav_despar'],
               'RendDP':['d_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'T_despar','RSO_despar', 'Lav_despar'],
               'd_DP':['IV_C', 'd_C', 'IR_C',  'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IR_DP':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IV_DP':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', 'Lav_despar']}



cv = KFold(n_splits=5, shuffle=True, random_state=358)

NN=MLPRegressor(hidden_layer_sizes=5, activation='tanh', solver='lbfgs', 
               alpha=0.01, batch_size=90, max_iter=500, 
                shuffle=True, random_state=247,  n_iter_no_change=20 )
#NN=MLPRegressor()

pipe=Pipeline([('scaler',StandardScaler()),('mod',NN)])
X_dp=dados_dp.loc[:,x_var_dp]

scores_collect=[]
for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(pipe,X_dp.loc[bool3,dict_feature[i]] , y2mod.loc[bool3],
                            cv=cv,scoring=['r2','neg_mean_squared_error'])
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.median(np.sqrt(-scores['test_neg_mean_squared_error'])))
  #  print(np.sum(1-bool3))

for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+3*y2mod.std(),y2mod>y2mod.mean()-3*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_val_score(clf,X_dp.loc[bool3,:] , y2mod.loc[bool3], cv=cv)
    scores_collect.append([i,scores])
    print(i,np.mean(scores),np.median(scores))

    

#%%  
for train_index, test_index in kf.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]


X_dp2=dados_dp.loc[:,x_var_dp].copy()
#X_dp2['TRSO2']=X_dp2['T_despar']*(X_dp2['RSO_despar'])
X_dp2['RSO2']=(X_dp2['RSO_desaro'])**2
#X_dp2['T2']=np.log(X_dp2['T_despar']+20)

scores_collect=[]
for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_val_score(clf,X_dp2.loc[bool3,:] , np.log(y2mod.loc[bool3]), cv=cv)
    scores_collect.append([i,scores])
    print(i,np.mean(scores),np.median(scores))
#    plt.figure()
#    y2mod.loc[bool3].plot.kde()
#    plt.title(i)
#    print(np.sum(1-bool3))
#%% feature selection
 
    
dict_feature={'IV_R':['IV_C', 'd_C', 'IR_C', 'RSO_desaro', 'Lav_despar'],
               'd_R':['IV_C', 'd_C', 'IR_C', 'RSO_desaro', 'RSO_despar', 'Lav_despar'], 
               'IR_R':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'RendR':[ 'IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', 'Lav_despar'],
               'RendDP':['d_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'T_despar','RSO_despar', 'Lav_despar'],
               'd_DP':['IV_C', 'd_C', 'IR_C',  'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IR_DP':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IV_DP':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', 'Lav_despar']}

scores_collect=[]
for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_val_score(clf,X_dp2.loc[bool3,dict_feature[i]] , np.log(y2mod.loc[bool3]), cv=cv)
    scores_collect.append([i,scores])
    print(i,np.mean(scores),np.median(scores))



X_dp=dados_dp.loc[:,x_var_dp]

for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_val_score(pipe,X_dp.loc[bool3,dict_feature[i]] , np.log(y2mod.loc[bool3]), cv=cv)
    scores_collect.append([i,scores])
    print(i,np.mean(scores),np.median(scores))
    
    
#%% try neural networks

X_dp=dados_dp.loc[:,x_var_dp]

for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_val_score(pipe,X_dp.loc[bool3,:] , np.log(y2mod.loc[bool3]), cv=cv)
    scores_collect.append([i,scores])
    print(i,np.mean(scores),np.median(scores))





