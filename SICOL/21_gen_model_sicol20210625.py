# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:45:44 2021

@author: Felipo Soares
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.linear_model import LinearRegression, RidgeCV,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pickle import dump

dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao_20210531.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao_20210601.csv')

dados_tot=pd.concat((dados_da,dados_dp))
dados_tot.drop(['T.1','RSO.1'],axis=1,inplace=True)
dados_tot=dados_tot.loc[dados_tot['Tipo'] != 'DAO',:]
dados_dp=dados_dp.iloc[:,1:]
x_var_da=['T_desaro', 'RSO_desaro','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_R', 'd_R', 'IR_R', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]


y_var_dp=[ 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']

from sklearn.ensemble import GradientBoostingRegressor
GBR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )

#%% modelo residual
cv = KFold(n_splits=5, shuffle=True, random_state=358)

X_dp=dados_tot.loc[:,x_var_dp]


x_var_dp=[  'd_R',  'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']

X_dp=dados_tot.loc[:,x_var_dp]

path_model=r'E:\coppe\SICOL_novo\modelos\20210625'
for i in y_var_dp:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           (np.sum(np.isnan(X_dp),axis=1)==0).values)
    bool3=np.logical_and(bool1,bool2)
    GBR.fit(X_dp.loc[bool3,:] , y2mod.loc[bool3])
    dump(GBR, open(path_model+'\\GBR20210625_' + i+ '.joblib','wb')) 

X_da=dados_tot.loc[:,x_var_da]

for i in y_var_da:
    
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           (np.sum(np.isnan(X_da),axis=1)==0).values)
    bool3=np.logical_and(bool1,bool2)
    GBR.fit(X_da.values[bool3,:] , y2mod.loc[bool3])
    dump(GBR, open(path_model+'\\GBR20210625_' + i+ '.joblib','wb'))  
    
y2mod=dados_tot.loc[:,'d_DP']
bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                       np.logical_not(np.sum(np.isnan(X_dp),axis=1)>=1).values)
bool3=np.logical_and(bool1,bool2)
X2mod=dados_tot.loc[bool3,x_var_dp].reset_index(drop=True)
y2mod=dados_tot.loc[bool3,'d_DP'].reset_index(drop=True)
feat_imp=[]
yhat_collect=np.zeros(y2mod.shape[0])
for train_index, test_index in cv.split(X2mod):
    X_train, X_test = X2mod.iloc[train_index,:], X2mod.iloc[test_index,:]
    y_train, y_test = y2mod[train_index], y2mod[test_index]
    clf.fit(X_train['d_R'].values.reshape(-1, 1),y_train)
    yhat_lin=clf.predict(X_train['d_R'].values.reshape(-1, 1))
    res_train=y_train-yhat_lin
    GBR.fit(X_train,res_train)
    yhat_GBR=GBR.predict(X_test)
    yhat_lin=clf.predict(X_test['d_R'].values.reshape(-1, 1))
    
    yhat=yhat_lin+yhat_GBR
    yhat_collect[test_index]=yhat
#    print(r2_score(y_test,yhat),np.sqrt(np.mean((y_test-yhat)**2)))
    print(r2_score(y_test-yhat_lin,yhat_GBR),np.sqrt(np.mean((y_test-yhat_lin-yhat_GBR)**2)))
    
    feat_imp.append(GBR.feature_importances_)
print(r2_score(y2mod,yhat_collect),np.sqrt(np.mean((y2mod-yhat_collect)**2)))

plt.figure();plt.plot(y2mod,yhat_collect,'o')
plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
plt.xlabel('d_DP real')
plt.ylabel('d_DP predito')

l=np.random.randn(100)
l1=l+np.random.randn(100)*0.1
print(np.corrcoef(l,l1))
plt.figure();plt.plot(l,l1,'o')

feat_imp=np.asarray(feat_imp)
plt.figure();ax=sns.barplot(data=feat_imp,ci='sd')
ticks=ax.get_xticklabels()
x_labels = list(ticks)
plt.xticks(ticks=np.arange(X_dp.columns.shape[0]),labels=X_dp.columns)
plt.title('residuo, d_DP')

#%% bow for IR
X2mod=dados_tot.loc[bool3,x_var_dp].reset_index(drop=True)
y2mod=dados_tot.loc[bool3,'IR_DP'].reset_index(drop=True)
feat_imp=[]
yhat_collect=np.zeros(y2mod.shape[0])

for train_index, test_index in cv.split(X2mod):
    X_train, X_test = X2mod.iloc[train_index,:], X2mod.iloc[test_index,:]
    y_train, y_test = y2mod[train_index], y2mod[test_index]
    clf.fit(X_train['IR_R'].values.reshape(-1, 1),y_train)
    res_train=y_train-clf.predict(X_train['IR_R'].values.reshape(-1, 1))
    GBR.fit(X_train,res_train)
    yhat=clf.predict(X_test['IR_R'].values.reshape(-1, 1))+GBR.predict(X_test)
    yhat_collect[test_index]=yhat
    print(r2_score(y_test,yhat),np.sqrt(np.mean((y_test-yhat)**2)))
    feat_imp.append(GBR.feature_importances_)
plt.figure();plt.plot(y2mod,yhat_collect,'o')
plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
plt.xlabel('IR_DP real')
plt.ylabel('IR_DP predito')
print(r2_score(y2mod,yhat_collect),np.sqrt(np.mean((y2mod-yhat_collect)**2)))

feat_imp=np.asarray(feat_imp)
plt.figure();ax=sns.barplot(data=feat_imp,ci='sd')
ticks=ax.get_xticklabels()
x_labels = list(ticks)
plt.xticks(ticks=np.arange(X_dp.columns.shape[0]),labels=X_dp.columns)
plt.title('residuo, IR_DP')
