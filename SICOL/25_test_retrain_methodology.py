# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:22:30 2021

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
from sklearn.ensemble import GradientBoostingRegressor

dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao_20210713.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao_20210713.csv')
dados_tot=pd.concat((dados_dp,dados_da))
dados_tot.drop('Unnamed: 0',inplace=True,axis=1)
dados_tot.loc[dados_tot['IV_R']<30,'IV_R']=np.nan
dados_tot.loc[dados_tot['IV_R']>200,'IV_R']=np.nan
dados_tot.loc[dados_tot['T_despar']>-1,'T_despar']=np.nan
dados_tot.loc[dados_tot['RendDP']<20,'RendDP']=np.nan
dados_tot.loc[dados_tot['Lav_despar']<1,'Lav_despar']=np.nan
dados_tot.reset_index(inplace=True,drop=True)

GBR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )

x_var_dp=[ 'IV_C', 'd_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_op=[ 'IV_C', 'd_C', 'IR_C','T_desaro', 'RSO_desaro', 'T_despar',
          'RSO_despar', 'Lav_despar',]

ind_inicial=np.arange(0,73,dtype=int)
ind_novos1=np.arange(73,100,dtype=int)
ind_novos2=np.arange(100,122,dtype=int)
y_var_dp=[ 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']
X_dp=dados_tot.loc[:,x_var_dp]
X_dp2=X_dp.copy()

def check_min_max_nan(X_prime,X_test,var_int,flag_both=False):
    maximo=X_prime.max().loc[var_int]
    minimo=X_prime.min().loc[var_int]
    
    flag_max=X_test.loc[:,var_int]<maximo
    flag_min=X_test.loc[:,var_int]>minimo
    flag_nan=np.logical_not(np.sum(np.isnan(X_test.loc[:,var_int]),axis=1)>=1).values
    flag_sample=np.logical_and(np.logical_or(flag_max.all(axis=1),flag_min.all(axis=1)),
                                flag_nan)
    
    if flag_both:
        return X_test.loc[flag_sample,:],(flag_sample,flag_max,flag_min)
    else:
        return X_test.loc[flag_sample,:]

def check_covar(mean_vec,cov_mat,X_test,flag_both=False,threshold=3):
    maha=np.zeros(X_test.shape[0])
    X_num=X_test.values
    
    for i in range(X_test.shape[0]):
        maha[i]=np.matmul(np.matmul((X_num[i,:]-mean_vec),np.linalg.inv(cov_mat)),
            (X_num[i,:]-mean_vec).T)
    flag_sample=np.logical_or(maha<threshold,maha>-threshold)
    
    if flag_both:
        return X_test.loc[flag_sample,:],flag_sample
    else:
        return X_test.loc[flag_sample,:]

GBR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )

dado_inicial=dados_tot.loc[ind_inicial,:]
X_inicial=dados_tot.loc[ind_inicial,x_var_dp]
X_novos1=dados_tot.loc[ind_novos1,x_var_dp]
dado_novo1=dados_tot.loc[ind_novos1,:]
X_novos2=dados_tot.loc[ind_novos2,x_var_dp]
dado_novo2=dados_tot.loc[ind_novos2,:]

for i in y_var_dp:
    
    y2mod=dado_inicial.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_inicial),axis=1)>=1).values),
                            np.logical_not(np.sum(X_inicial==0,axis=1)>=1).values)
    
    bool3=np.logical_and(bool1,bool2)
    dado_inicial_limpo=dado_inicial.loc[bool3,:].reset_index(drop=True)
    X2mod=X_inicial.loc[bool3,:].reset_index(drop=True)
    Cov=X2mod.cov().loc[x_var_dp,x_var_dp]
    Mean=X2mod.mean().loc[x_var_dp]
    y2mod=dado_inicial.loc[bool3,i].reset_index(drop=True)
    GBR.fit(X2mod,y2mod)
    # Novos dados 1
    dado_novo=check_min_max_nan(dado_inicial,dado_novo1,x_var_dp)
    print('perda de dados min_max',
          dado_novo.shape[0]-dado_novo1.shape[0])
    dado_novo=check_covar(Mean,Cov,dado_novo.loc[:,x_var_dp])
    
    print('perda de dados cov',
          dado_novo.shape[0]-dado_novo1.shape[0])
    ind1=dado_novo.index
    y2mod1=dado_novo1.loc[ind1,i]
    X2mod1=dado_novo
    yhat1=GBR.predict(X2mod1)
    erro=np.sqrt(np.mean((y2mod1-yhat1)**2))
    print(i,erro)
    # Novos dados 2
    dado_2=pd.concat((dado_inicial_limpo,pd.concat((dado_novo,y2mod1),axis=1)))  
    X2mod=dado_2.loc[:,x_var_dp].reset_index(drop=True)

    Cov=X2mod.cov().loc[x_var_dp,x_var_dp]
    Mean=X2mod.mean().loc[x_var_dp]
    y2mod=dado_2.loc[:,i].reset_index(drop=True)
    GBR.fit(X2mod,y2mod)
    dado_novo=check_min_max_nan(dado_2,dado_novo2,x_var_dp)
    print('perda de dados min_max retreino2',
          dado_novo.shape[0]-dado_novo2.shape[0])
    dado_novo=check_covar(Mean,Cov,dado_novo2.loc[:,x_var_dp])
    
    print('perda de dados cov retreino2',
          dado_novo.shape[0]-dado_novo2.shape[0])
    ind2=dado_novo.index
    y2mod1=dado_novo2.loc[ind2,i]
    X2mod1=dado_novo
    yhat1=GBR.predict(X2mod1)
    erro=np.sqrt(np.mean((y2mod1-yhat1)**2))
    print(' retreino2',i,erro)
    
    
#    yhat_collect=np.zeros(y2mod.shape[0])
#
#
#    plt.figure();plt.plot(y2mod,yhat_collect,'o')
#    plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
#    plt.xlabel(i+' real')
#    plt.ylabel(i+' predito')


#mean=np.asarray([10,20,30])
#cov=np.asarray([[1,1,-3],[-2,1/3,1],[1/4,5,1/6]])
#cov=cov*cov.T
#x = np.random.multivariate_normal(mean, cov, 1000)
#k=np.zeros(1000)
#for i in range(1000):
#    k[i]=np.matmul(np.matmul((x[i,:]-mean),np.linalg.inv(cov)),(x[i,:]-mean).T)
#    
#plt.figure();plt.hist(k,bins=30)
#np.linalg.det(cov)
def corrfromCov(cov):
    std=np.sqrt(np.diag(cov))
    for i in range(std.shape[0]):
        for j in range(std.shape[0]):
            cov[i,j]=cov[i,j]/std[i]/std[j]
    return cov 