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


def check_min_max_nan(new_data,old_data,var_int,flag_both=False):
    maximo=old_data.max().loc[var_int]
    minimo=old_data.min().loc[var_int]
    
    flag_max=new_data.loc[:,var_int]<maximo
    flag_min=new_data.loc[:,var_int]>minimo
    flag_nan=np.logical_not(np.sum(np.isnan(new_data.loc[:,var_int]),axis=1)>=1).values
    flag_sample=np.logical_and(np.logical_or(flag_max.all(axis=1),flag_min.all(axis=1)),
                                flag_nan)
    
    if flag_both:
        return new_data.loc[flag_sample,:],new_data.loc[np.logical_not(flag_sample),:],(flag_sample,flag_max,flag_min)
    else:
        return new_data.loc[flag_sample,:],new_data.loc[np.logical_not(flag_sample),:]

def check_covar(mean_vec,cov_mat,X_test,var,flag_both=False,threshold=10):
    maha=np.zeros(X_test.shape[0])
    X_num=X_test.loc[:,var].values
    
    for i in range(X_test.shape[0]):
        maha[i]=np.matmul(np.matmul((X_num[i,:]-mean_vec),np.linalg.inv(cov_mat)),
            (X_num[i,:]-mean_vec).T)
    flag_sample=np.sqrt(np.abs(maha))<threshold
    
    if flag_both:
        return X_test.loc[flag_sample,:],X_test.loc[np.logical_not(flag_sample),:],flag_sample
    else:
        return X_test.loc[flag_sample,:],X_test.loc[np.logical_not(flag_sample),:]

def retrain(new_data,old_data,model,variables,target,error_threshold=5
            ,cov_threshold=10, check_target=False,flag_retrain=False):
    '''
    new_data são os dados novos
    old_data são os dados antigos
    model é o modelo utilizado
    variables são as variáveis de entrada a serem analisadas
    target é a variável de saída
    error_threshold é o limite do erro que dispara o retreinamento
    
    cov_threshold é o limite entre a distância dos dados antigos apra os novos
    check_target analisa se deve-se analisar a variável de entrada também
    
    
    model é o m
    reserve_dataset é o dataset reserva para dados muito diferentes
    
    
    '''
    reserve_dataset=pd.DataFrame()
    if check_target:
        var=variables.copy()
        var.append(target)
    else:
        var=variables.copy()
    Cov=old_data.cov().loc[var,var]
    Mean=old_data.mean().loc[var]

    dado_novo,dado_storage=check_min_max_nan(new_data,old_data,var)
    reserve_dataset=pd.concat((reserve_dataset,dado_storage))
    dado_novo,dado_storage=check_covar(Mean,Cov,dado_novo,var)
    reserve_dataset=pd.concat((reserve_dataset,dado_storage))
    ind1=dado_novo.index
    y2mod=new_data.loc[ind1,target]
    if check_target:
        X2mod=dado_novo.loc[:,variables].copy()    
        dados_novos_DF=pd.DataFrame(dado_novo,columns=var)
        new_dataset=pd.concat((old_data,dados_novos_DF))
    else:
        X2mod=dado_novo.copy()
        dados_novos_DF=pd.DataFrame(dado_novo,columns=var)
        target_novo_DF=pd.DataFrame(y2mod,columns=target)
        dados_novos_DF=pd.concat((dados_novos_DF,target_novo_DF),axis=1)
        new_dataset=pd.concat((old_data,dados_novos_DF))
    
    yhat=model.predict(X2mod)
    erro=np.sqrt(np.mean((y2mod-yhat)**2))

    if erro>error_threshold and not flag_retrain:
        print('Recomendo treino')
    elif erro>error_threshold and flag_retrain:
        X2mod=new_dataset.loc[:,variables]
        y2mod=new_dataset.loc[:,target]
        model.fit(X2mod,y2mod)
        
        
    return model, new_dataset,reserve_dataset


#%%Exemplo
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
dado_rej_collect_list=[]
for i in y_var_dp:
    y2mod=dado_inicial.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_inicial),axis=1)>=1).values),
                            np.logical_not(np.sum(X_inicial==0,axis=1)>=1).values)
    
    bool3=np.logical_and(bool1,bool2)
    dado_inicial_limpo=dado_inicial.loc[bool3,:].reset_index(drop=True)
    X2mod=X_inicial.loc[bool3,:].reset_index(drop=True)
    y2mod=dado_inicial.loc[bool3,i].reset_index(drop=True)
    GBR.fit(X2mod,y2mod)
    # Novos dados 1
    GBR, new_DS,reserve_DS = retrain(dado_novo1,dado_inicial_limpo,GBR,x_var_dp,i,
            check_target=True)
    
    GBR, new_DS,reserve_DS = retrain(dado_novo2,new_DS,GBR,x_var_dp,i,
            check_target=True)
    dado_rej_collect_list.append(reserve_DS)

    # Novos dados 2
    
#    print('perda de dados cov retreino2',

#%% Deasromizacao
    
x_var_da=[ 'd_C',  'T_desaro', 'RSO_desaro']

ind_inicial1=np.arange(0,73,dtype=int)
ind_inicial2=np.arange(122,231,dtype=int)
ind_inicial=np.hstack((ind_inicial1,ind_inicial2))
ind_novos1=np.arange(73,100,dtype=int)
ind_novos2a=np.arange(100,122,dtype=int)
ind_novos2b=np.arange(231,267,dtype=int)
ind_novos2=np.hstack((ind_novos2a,ind_novos2b))

y_var_da=[ 'RendR', 'd_R', 'IR_R', 'IV_R']
X_dp=dados_tot.loc[:,x_var_da]
X_dp2=X_dp.copy()
dado_inicial=dados_tot.loc[ind_inicial,:]
X_inicial=dados_tot.loc[ind_inicial,x_var_da]
X_novos1=dados_tot.loc[ind_novos1,x_var_da]
dado_novo1=dados_tot.loc[ind_novos1,:]
X_novos2=dados_tot.loc[ind_novos2,x_var_da]
dado_novo2=dados_tot.loc[ind_novos2,:]

for i in y_var_da:
    dado_rej_collect=pd.DataFrame()
    x_var_da2=x_var_da.copy()
    x_var_da2.append(i)
    y2mod=dado_inicial.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_inicial),axis=1)>=1).values),
                            np.logical_not(np.sum(X_inicial==0,axis=1)>=1).values)
    
    bool3=np.logical_and(bool1,bool2)
    dado_inicial_limpo=dado_inicial.loc[bool3,:].reset_index(drop=True)
    X2mod=X_inicial.loc[bool3,:].reset_index(drop=True)
    Cov=X2mod.cov().loc[x_var_da,x_var_da]
    Mean=X2mod.mean().loc[x_var_da]
    y2mod=dado_inicial.loc[bool3,i].reset_index(drop=True)
    GBR.fit(X2mod,y2mod)
    # Novos dados 1
    GBR, new_DS,reserve_DS = retrain(dado_novo1,dado_inicial_limpo,GBR,x_var_da,i,
            check_target=True)
    
    GBR, new_DS,reserve_DS = retrain(dado_novo2,new_DS,GBR,x_var_da,i,
            check_target=True)
    dado_rej_collect_list.append(reserve_DS)

    
