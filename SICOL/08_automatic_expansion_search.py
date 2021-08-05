# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:58:34 2021

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

import itertools
n = 15
comb=[]
for n1 in range(3,n+1):
    for x in itertools.combinations( range(n), n1 ) :
        comb.append([ True if i in x else False for i in range(n) ])
    
dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao.csv')
dados_dp=dados_dp.iloc[:,1:]
x_var_da=['T', 'RSO','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_C',  'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar']
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
data_collect=[]

for i in comb:
    X=X_da2.loc[:,i].copy()
    cols=X_da2.columns[i]
    model_esp=' + '.join(cols)    
    X = sm.add_constant(X)
    collect1=[]
    for j in y_var_da:
        y2mod=dados_da.loc[:,j]
        bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())

        mod = sm.OLS(y2mod.loc[bool1],X.loc[bool1,:], missing='drop')
        res=mod.fit()
        collect1.append([j,res.bic])
    data_collect.append([model_esp,collect1])

data_numpy=np.asarray(data_collect)

formulas=data_numpy[:,0]
results=data_numpy[:,1]
DF_results=np.asarray(list(results))
DF_results2=pd.DataFrame(DF_results[:,:,1])
DF_results2.columns=y_var_da
DF_results2.index=formulas
DF_results3=DF_results2.astype(float)
print(DF_results3.idxmin())

#%%
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
data_collect=[]
n = 23
comb=[]
for n1 in range(4,n+1):
    for x in itertools.combinations( range(n), n1 ) :
        comb.append([ True if i in x else False for i in range(n) ])
picks= np.random.choice(len(comb), size=20000, replace=False, p=None)
for i in range(len(picks)):
    X=X_dp2.loc[:,comb[picks[i]]].copy()
    cols=X_dp2.columns[comb[picks[i]]]
    model_esp=' + '.join(cols)    
    X = sm.add_constant(X)
    collect1=[]
    for j in y_var_dp:
        y2mod=dados_dp.loc[:,j]
        bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
        mod = sm.OLS(y2mod.loc[bool1],X.loc[bool1,:], missing='drop')
        res=mod.fit()
        collect1.append([j,res.bic])
    data_collect.append([model_esp,collect1])
    if np.random.rand(1)>0.9995:
        print(i)
        
data_numpy=np.asarray(data_collect)

formulas=data_numpy[:,0]
results=data_numpy[:,1]
DF_results=np.asarray(list(results))
DF_results2=pd.DataFrame(DF_results[:,:,1])
DF_results2.columns=y_var_dp
DF_results2.index=formulas
DF_results3=DF_results2.astype(float)
print(DF_results3.idxmin())
a=DF_results3.idxmin()



