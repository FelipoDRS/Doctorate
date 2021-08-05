# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:59:47 2021

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
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.linear_model import LinearRegression, RidgeCV,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao.csv')
dados_dp=dados_dp.iloc[:,1:]
x_var_da=['T', 'RSO','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_C', 'd_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]


y_var_dp=['IV_R', 'd_R', 'IR_R','RendR', 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']

cv = KFold(n_splits=5, shuffle=True, random_state=358)

list_exp=['log','inv','exp','sqrt','square','inter']
list_xvar=['T','RSO']
def apply_exp (data,expansion):
    if expansion == 'log':
        return np.log(data)
    elif expansion == 'sqrt':
        return np.sqrt(data)
    elif expansion == 'inv':
        return 1/data
    elif expansion == 'square':
        return data*data
    elif expansion == 'exp':
        return np.exp(data/10)
    elif expansion == 'inter':
        return data.iloc[:,0]*data.iloc[:,1]
    else:
        print('expansion not defined')

dict_features={'IV_R':	['d_C','inv_RSO','sqrt_RSO','inter_RSO','inter_T_logRSO'],
'd_R':['d_C','sqrt_RSO','inter_RSO','inter_T_logRSO'],
'IR_R':	['d_C','sqrt_RSO','inter_RSO','inter_T_logRSO'],
'RendR':	['T','RSO','d_C','square_T','inter_T_logRSO'],}

X_da=dados_da.loc[:,x_var_da]
X_da2=X_da.copy()
for i in list_exp:
    if i == 'inter':
        temp=apply_exp(X_da2[list_xvar],i)
        temp.name=i+'_'+j
        X_da2=pd.concat([X_da2,temp],axis=1)
    else:
        for j in list_xvar:
            temp=apply_exp(X_da2[j],i)
            temp.name=i+'_'+j
            X_da2=pd.concat([X_da2,temp],axis=1)
#    print(X_da2.columns)
X_da2['inter_T_logRSO']=X_da2['T']*X_da2['log_RSO']            

X_da=dados_da.loc[:,x_var_da]
clf = LinearRegression()
scores_collect=[]
for i in y_var_da:
    y2mod=dados_da.loc[:,i]
    boll1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    scores = cross_validate(clf,X_da2.loc[boll1,dict_features[i]] , y2mod.loc[boll1], cv=cv,scoring=['r2','neg_mean_squared_error'])
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.median(np.sqrt(-scores['test_neg_mean_squared_error'])))


dict_deatures_dp={'IV_R':	['IV_C','T_desaro','T_despar','RSO_despar','log_T_desaro','sqrt_RSO_desaro','square_T_despar','inter_T_desaro_T_despar','inter_RSO_desaro_T_despar','inter_T_despar_RSO_despar'],
'd_R':	['IR_C','T_desaro','RSO_desaro','log_RSO_despar','sqrt_RSO_despar','square_T_desaro','inter_T_desaro_RSO_despar'],
'IR_R':	['IR_C','T_desaro','RSO_desaro','log_RSO_despar','sqrt_RSO_despar','square_T_desaro','inter_T_desaro_RSO_despar'],
'RendR':	['IV_C','IR_C','T_desaro','RSO_despar','sqrt_RSO_despar','square_T_desaro','square_RSO_desaro'],
'RendDP':	['IV_C','log_T_desaro','inter_T_desaro_RSO_despar','inter_RSO_desaro_RSO_despar','inter_T_despar_RSO_despar'],
'd_DP':	['IR_C','T_desaro','RSO_desaro','log_RSO_despar','sqrt_RSO_despar','square_T_desaro','inter_T_desaro_RSO_despar'],
'IR_DP':	['IV_C','IR_C','T_desaro','RSO_desaro','log_T_desaro','sqrt_T_desaro','sqrt_RSO_despar','square_RSO_despar'],
'IV_DP':	['IV_C','IR_C','log_RSO_despar','sqrt_RSO_desaro','inter_T_desaro_RSO_despar','inter_T_logRSO'],
}

list_exp=['log','sqrt','square','inter']

list_xvar=['T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar']

X_dp=dados_dp.loc[:,x_var_dp]

X_dp2=X_dp.copy()
#list_xvar=['T','RSO']

for i in list_exp:
    if i != 'inter':
        for j in list_xvar:
            if (i =='log' or i =='sqrt') and j== 'T_despar':
                pass
            else:
                temp=apply_exp(X_dp2[j],i)
                temp.name=i+'_'+j
                X_dp2=pd.concat([X_dp2,temp],axis=1)
    else:
        for j in range(len(list_xvar)):
            for jj in range(j,len(list_xvar)):
                if j != jj:
                    temp=apply_exp(X_dp2[[list_xvar[j],list_xvar[jj]]],i)
                    temp.name=i+'_'+list_xvar[j]+'_'+list_xvar[jj]
                    X_dp2=pd.concat([X_dp2,temp],axis=1)
#    print(X_da2.columns)
X_dp2['inter_T_logRSO']=X_dp2['T_desaro']*X_dp2['log_RSO_desaro']            


scores_collect=[]
for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(clf,X_dp2.loc[bool3,dict_deatures_dp[i]] , y2mod.loc[bool3], 
                            cv=cv,scoring=['r2','neg_mean_squared_error'])
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.mean(np.sqrt(-scores['test_neg_mean_squared_error'])))
#    print(np.sum(1-bool3))