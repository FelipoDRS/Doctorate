# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:17:33 2021

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

dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao_20210531.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao_20210601.csv')
dados_tot=pd.concat((dados_da,dados_dp))
dados_tot.reset_index(inplace=True,drop=True)
dados_tot.drop(['T.1','RSO.1'],axis=1,inplace=True)
dados_dp=dados_dp.iloc[:,1:]
x_var_da=['T_desaro', 'RSO_desaro','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=['d_C', 'IV_R', 'd_R', 'IR_R','IV_Rhat', 'd_Rhat', 'IR_Rhat', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]


y_var_dp=[ 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']

cv = KFold(n_splits=5, shuffle=True, random_state=358)

list_exp=['log','inv','exp','sqrt','square','inter']
list_xvar=['T_desaro','RSO_desaro']
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

dict_features={'IV_R':	['d_C','inv_RSO_desaro','sqrt_RSO_desaro','inter_T_desaro_RSO_desaro',
                        'inter_T_logRSO_desaro'],
'd_R':['d_C','log_RSO_desaro','exp_T_desaro','inter_T_desaro_RSO_desaro'],
'IR_R':	['d_C','log_RSO_desaro','exp_T_desaro','inter_T_desaro_RSO_desaro'],
'RendR':	['d_C','log_RSO_desaro','inter_T_logRSO_desaro']}



X_da=dados_tot.loc[:,x_var_da]
X_da2=X_da.copy()
for i in list_exp:
    if i == 'inter':
        temp=apply_exp(X_da2[list_xvar],i)
        temp.name='inter_T_desaro_RSO_desaro'
        X_da2=pd.concat([X_da2,temp],axis=1)
    else:
        for j in list_xvar:
            temp=apply_exp(X_da2[j],i)
            temp.name=i+'_'+j
            X_da2=pd.concat([X_da2,temp],axis=1)
#    print(X_da2.columns)
X_da2['inter_T_logRSO_desaro']=X_da2['T_desaro']*X_da2['log_RSO_desaro']            

X_da=dados_tot.loc[:,x_var_da]
clf_dR = LinearRegression()
clf_IVR = LinearRegression()
clf_IRR = LinearRegression()
scores_collect=[]

i='d_R'
y2mod=dados_tot.loc[:,i]
bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                       np.logical_not(np.sum(np.isnan(X_da2),axis=1)>=1).values)
bool3=np.logical_and(bool1,bool2)
clf_dR.fit(X_da2.loc[bool3,dict_features[i]] , y2mod.loc[bool3])
dados_tot['d_Rhat']=0
dados_tot.loc[bool3,'d_Rhat']=clf_dR.predict(X_da2.loc[bool3,dict_features[i]])


i='IV_R'

y2mod=dados_tot.loc[:,i]
bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                       np.logical_not(np.sum(np.isnan(X_da2),axis=1)>=1).values)
bool3=np.logical_and(bool1,bool2)
clf_IVR.fit(X_da2.loc[bool3,dict_features[i]] , y2mod.loc[bool3])
dados_tot['IV_Rhat']=0
dados_tot.loc[bool3,'IV_Rhat']=clf_IVR.predict(X_da2.loc[bool3,dict_features[i]])
i='IR_R'

y2mod=dados_tot.loc[:,i]
bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                       np.logical_not(np.sum(np.isnan(X_da2),axis=1)>=1).values)
bool3=np.logical_and(bool1,bool2)
clf_IRR.fit(X_da2.loc[bool3,dict_features[i]] , y2mod.loc[bool3])
dados_tot['IR_Rhat']=0
dados_tot.loc[bool3,'IR_Rhat']=clf_IRR.predict(X_da2.loc[bool3,dict_features[i]])

#%% desparafinacao
dict_features_dp={'RendDP':	['IR_R','T_desaro','sqrt_T_desaro','square_T_desaro','square_RSO_desaro','square_T_despar','square_RSO_despar'],
'd_DP':	['IV_R','IR_R','T_desaro','RSO_despar','log_T_desaro','sqrt_T_desaro','sqrt_RSO_desaro','square_T_desaro','square_RSO_desaro','inter_T_desaro_RSO_desaro','inter_T_desaro_RSO_despar','inter_RSO_desaro_RSO_despar'],
'IR_DP':	['IV_R','IR_R','T_desaro','log_T_desaro','sqrt_T_desaro','sqrt_RSO_desaro','square_T_desaro','inter_T_desaro_RSO_desaro','inter_T_desaro_RSO_despar','inter_RSO_desaro_RSO_despar','inter_T_logRSO_desaro'],
'IV_DP':	['IR_R','T_desaro','RSO_despar','log_T_desaro','log_RSO_desaro','sqrt_T_desaro','square_T_desaro','square_RSO_desaro','inter_RSO_desaro_RSO_despar']}


dict_features_dphat={'RendDP':	['IR_Rhat','T_desaro','sqrt_T_desaro','square_T_desaro','square_RSO_desaro','square_T_despar','square_RSO_despar'],
'd_DP':	['IV_Rhat','IR_Rhat','T_desaro','RSO_despar','log_T_desaro','sqrt_T_desaro','sqrt_RSO_desaro','square_T_desaro','square_RSO_desaro','inter_T_desaro_RSO_desaro','inter_T_desaro_RSO_despar','inter_RSO_desaro_RSO_despar'],
'IR_DP':	['IV_Rhat','IR_Rhat','T_desaro','log_T_desaro','sqrt_T_desaro','sqrt_RSO_desaro','square_T_desaro','inter_T_desaro_RSO_desaro','inter_T_desaro_RSO_despar','inter_RSO_desaro_RSO_despar','inter_T_logRSO_desaro'],
'IV_DP':	['IR_Rhat','T_desaro','RSO_despar','log_T_desaro','log_RSO_desaro','sqrt_T_desaro','square_T_desaro','square_RSO_desaro','inter_RSO_desaro_RSO_despar']}



list_exp=['log','inv','exp','sqrt','square','inter']

list_xvar=['T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar']

X_dp=dados_tot.loc[:,x_var_dp]
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
X_dp2['inter_T_logRSO_desaro']=X_dp2['T_desaro']*X_dp2['log_RSO_desaro']            
X_da_novos=X_dp2.iloc[:144,:]
cv = KFold(n_splits=5, shuffle=True, random_state=358)
clf=LinearRegression()
scores_collect=[]
SC=StandardScaler()
for i in y_var_dp:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values),
                            np.logical_not(np.sum(X_dp2==0,axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    X2mod=X_dp2.loc[bool3,:].reset_index(drop=True)
    y2mod=dados_tot.loc[bool3,i].reset_index(drop=True)
    yhat_collect=np.zeros(y2mod.shape[0])

    for train_index, test_index in cv.split(X2mod):
        X_train, X_test = X2mod.iloc[train_index,:], X2mod.iloc[test_index,:]
        y_train, y_test = y2mod[train_index], y2mod[test_index]
        clf.fit(X_train.loc[:,dict_features_dp[i]],y_train)
        
        yhat=clf.predict(X_test.loc[:,dict_features_dphat[i]])
        yhat_collect[test_index]=yhat
        #print(r2_score(y_test,yhat),np.sqrt(np.mean((y_test-yhat)**2)))
    print(i,r2_score(y2mod,yhat_collect),np.sqrt(np.mean((y2mod-yhat_collect)**2)))            
    plt.figure();plt.plot(y2mod,yhat_collect,'o')
    plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
    plt.xlabel(i+' real')
    plt.ylabel(i+' predito')

#%% metodologia 2
dados_tot.loc[:,['IR_Rhat','IV_Rhat','d_Rhat']]=0
for i in y_var_dp:
    
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values),
                            np.logical_not(np.sum(X_dp2==0,axis=1)>=1).values)
    
    bool3=np.logical_and(bool1,bool2)
    X2mod=X_dp2.loc[bool3,:].reset_index(drop=True)
    y2mod=dados_tot.loc[bool3,i].reset_index(drop=True)
    yhat_collect=np.zeros(y2mod.shape[0])

    for train_index, test_index in cv.split(X2mod):
        X2mod.loc[:,['IR_Rhat','IV_Rhat','d_Rhat']]=0

        X_train, X_test = X2mod.iloc[train_index,:], X2mod.iloc[test_index,:]
        y_train, y_test = y2mod[train_index], y2mod[test_index]
        X_train_da=X_train.copy();y_train_da=y_train.copy()
        X_train_da=pd.concat((X_da_novos,X_train_da),axis=0)
        y_train_da=pd.concat((X_da_novos[['IR_R','IV_R','d_R']],X_train.loc[:,['IR_R','IV_R','d_R']]),axis=0)
        clf_IRR.fit(X_train_da.loc[:,dict_features['IR_R']] , y_train_da[['IR_R']])    
        X_test['IR_Rhat']=clf_IRR.predict(X_test.loc[:,dict_features['IR_R']])
        clf_dR.fit(X_train_da.loc[:,dict_features['d_R']] , y_train_da[['d_R']])    
        X_test['d_Rhat']=clf_dR.predict(X_test.loc[:,dict_features['d_R']])
        boll4=np.sum(np.isnan(y_train_da),axis=0)

        clf_IVR.fit(X_train_da.loc[boll4,dict_features['IV_R']] , y_train_da.loc[boll4,['IV_R']])    
        X_test['IV_Rhat']=clf_IVR.predict(X_test.loc[:,dict_features['IV_R']])

        clf.fit(X_train.loc[:,dict_features_dp[i]],y_train)
        yhat=clf.predict(X_test.loc[:,dict_features_dphat[i]])
        yhat_collect[test_index]=yhat
        #print(r2_score(y_test,yhat),np.sqrt(np.mean((y_test-yhat)**2)))
    print(i,r2_score(y2mod,yhat_collect),np.sqrt(np.mean((y2mod-yhat_collect)**2)))            
    plt.figure();plt.plot(y2mod,yhat_collect,'o')
    plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
    plt.xlabel(i+' real')
    plt.ylabel(i+' predito')

#%% metodologia 2 gradient boosting
    
from sklearn.ensemble import GradientBoostingRegressor
GBR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )
GBR_IVR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )
GBR_dR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )
GBR_IRR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )

dados_tot.loc[:,['IR_Rhat','IV_Rhat','d_Rhat']]=0
for i in y_var_dp:
    
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values),
                            np.logical_not(np.sum(X_dp2==0,axis=1)>=1).values)
    
    bool3=np.logical_and(bool1,bool2)
    X2mod=X_dp2.loc[bool3,:].reset_index(drop=True)
    y2mod=dados_tot.loc[bool3,i].reset_index(drop=True)
    yhat_collect=np.zeros(y2mod.shape[0])

    for train_index, test_index in cv.split(X2mod):
        X2mod.loc[:,['IR_Rhat','IV_Rhat','d_Rhat']]=0

        X_train, X_test = X2mod.iloc[train_index,:], X2mod.iloc[test_index,:]
        y_train, y_test = y2mod[train_index], y2mod[test_index]
        X_train_da=X_train.copy();y_train_da=y_train.copy()
        X_train_da=pd.concat((X_da_novos,X_train_da),axis=0)
        y_train_da=pd.concat((X_da_novos[['IR_R','IV_R','d_R']],X_train.loc[:,['IR_R','IV_R','d_R']]),axis=0)
        GBR_IRR.fit(X_train_da.loc[:,dict_features['IR_R']] , y_train_da[['IR_R']])    
        X_test['IR_Rhat']=clf_IRR.predict(X_test.loc[:,dict_features['IR_R']])
        GBR_dR.fit(X_train_da.loc[:,dict_features['d_R']] , y_train_da[['d_R']])    
        X_test['d_Rhat']=clf_dR.predict(X_test.loc[:,dict_features['d_R']])
        boll4=np.sum(np.isnan(y_train_da),axis=0)

        GBR_IVR.fit(X_train_da.loc[boll4,dict_features['IV_R']] , y_train_da.loc[boll4,['IV_R']])    
        X_test['IV_Rhat']=clf_IVR.predict(X_test.loc[:,dict_features['IV_R']])

        GBR.fit(X_train.loc[:,dict_features_dp[i]],y_train)
        yhat=GBR.predict(X_test.loc[:,dict_features_dphat[i]])
        yhat_collect[test_index]=yhat
        #print(r2_score(y_test,yhat),np.sqrt(np.mean((y_test-yhat)**2)))
    print(i,r2_score(y2mod,yhat_collect),np.sqrt(np.mean((y2mod-yhat_collect)**2)))            
    plt.figure();plt.plot(y2mod,yhat_collect,'o')
    plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
    plt.xlabel(i+' real')
    plt.ylabel(i+' predito')
