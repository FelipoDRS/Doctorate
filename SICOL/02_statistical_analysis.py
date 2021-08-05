# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:13:12 2021

@author: Felipo Soares
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao.csv')
dados_dp=dados_dp.iloc[:,1:]
x_var_da=['T', 'RSO','d_C']

y_var_da=['Visc_60R',  'Visc100_R', 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_C', 'd_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]


y_var_dp=['IV_R', 'd_R', 'IR_R','RendR', 'RendDP', 'd_DP', 'IR_DP',  'PL_DP', 
       'IV_DP']


X=dados_dp.loc[:,x_var_dp]
X = sm.add_constant(X)
y=dados_dp.loc[:,y_var_dp]

#mod = sm.OLS(y.iloc[:,12],X, missing='drop')
#res=mod.fit();print(res.summary())

for i in y_var_dp:
   y2mod=y.loc[:,i]
   bool1=np.logical_and(y2mod<y2mod.mean()+3*y2mod.std(),y2mod>y2mod.mean()-3*y2mod.std())
 
   mod = sm.RLM(y2mod.loc[bool1],X.loc[bool1,:], missing='drop',M=sm.robust.norms.Hampel())
   res=mod.fit();print(res.summary())
   yhat=res.predict(X)
   real_ind=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X),axis=1)>=1).values)
   print('R2 = ' ,r2_score(y2mod.loc[real_ind],yhat.loc[real_ind]))

dict_feature={'IV_R':['const','IV_C', 'd_C', 'IR_C', 'RSO_desaro', 'Lav_despar'],
               'd_R':['const','IV_C', 'd_C', 'IR_C', 'RSO_desaro', 'RSO_despar', 'Lav_despar'], 
               'IR_R':['const','IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'RendR':['const', 'IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', 'Lav_despar'],
               'RendDP':['const','d_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'T_despar','RSO_despar', 'Lav_despar'],
               'd_DP':['const','IV_C', 'd_C', 'IR_C',  'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IR_DP':['const','IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IV_DP':['const','IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', 'Lav_despar']}


for i in y_var_dp:
   y2mod=y.loc[:,i]
   bool1=np.logical_and(y2mod<y2mod.mean()+3*y2mod.std(),y2mod>y2mod.mean()-3*y2mod.std())
 
   mod = sm.OLS(y2mod.loc[bool1],X.loc[bool1,dict_feature[i]], missing='drop')#,M=sm.robust.norms.Hampel())
   res=mod.fit();print(res.summary())
   yhat=res.predict(X.loc[:,dict_feature[i]])
   real_ind=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X),axis=1)>=1).values)
   print('R2 = ' ,r2_score(y2mod.loc[real_ind],yhat.loc[real_ind]))


dados_group=dados_dp.loc[np.logical_and(np.logical_and(dados_dp['Unnamed: 1']!='TL',
                                                       dados_dp['Unnamed: 1']!='DAO'),
    dados_dp['Unnamed: 1']!='SPM')]
X_group=dados_group.loc[:,x_var_dp]

for i in y_var_dp:
   y2mod=dados_group.loc[:,i]
   real_ind=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_group),axis=1)>=1).values)

    
   y2mod=dados_group.loc[real_ind,i]
   X2mod=X_group.loc[real_ind,:]
   group=dados_group.loc[real_ind,'Unnamed: 1']
   mod = sm.MixedLM(y2mod,X2mod, missing='drop',groups=group)
   res=mod.fit();print(res.summary())
   yhat=res.predict(X2mod)
   print('R2 = ' ,r2_score(y2mod,yhat))


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV,Ridge
from pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = LinearRegression()
boll1=y2mod<200
Ri=Ridge(0.000,fit_intercept=True)
X3mod=StandardScaler().fit_transform(X2mod.loc[boll1,:])
scores = cross_val_score(Ri,X3mod , y2mod.loc[boll1], cv=8)

RCV=RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=False, normalize=True,  cv=5       )
RCV.fit(X2mod.loc[boll1,:],y2mod.loc[boll1])


tipos=dados_dp["Unnamed: 1"].unique()
for i in tipos:
    CC=dados_dp.loc[dados_dp["Unnamed: 1"]==i,:].corr()
    CC2=CC.dropna(axis=0, how='all')
    CC2=CC2.dropna(axis=1, how='any')
    print(i,np.linalg.det(CC2))
