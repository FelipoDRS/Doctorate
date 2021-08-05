# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:30:35 2021

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
dados_tot.drop(['T.1','RSO.1'],axis=1,inplace=True)
dados_dp=dados_dp.iloc[:,1:]
x_var_da=['T_desaro', 'RSO_desaro','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_R', 'd_R', 'IR_R', 'T_desaro',
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

dict_features={'IV_R':	['d_C','inv_RSO_desaro','sqrt_RSO_desaro','inter_RSO_desaro',
                        'inter_T_logRSO_desaro'],
'd_R':['d_C','log_RSO_desaro','exp_T_desaro','inter_RSO_desaro'],
'IR_R':	['d_C','log_RSO_desaro','exp_T_desaro','inter_RSO_desaro'],
'RendR':	['d_C','log_RSO_desaro','inter_T_logRSO_desaro']}



X_da=dados_tot.loc[:,x_var_da]
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
X_da2['inter_T_logRSO_desaro']=X_da2['T_desaro']*X_da2['log_RSO_desaro']            

X_da=dados_tot.loc[:,x_var_da]
clf = LinearRegression()
scores_collect=[]

for i in y_var_da:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_da2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(clf,X_da2.loc[bool3,dict_features[i]] , y2mod.loc[bool3], cv=cv,scoring=['r2','neg_mean_squared_error'])
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.median(np.sqrt(-scores['test_neg_mean_squared_error'])))
#%% Puramente linear
for i in y_var_da:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_da),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(clf,X_da.loc[bool3,:] , y2mod.loc[bool3], cv=cv,scoring=['r2','neg_mean_squared_error'])
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.median(np.sqrt(-scores['test_neg_mean_squared_error'])))

#%% gradient boosting
from sklearn.ensemble import GradientBoostingRegressor
GBR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )


scores_collect=[]
for i in y_var_da:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_da2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(GBR,X_da.loc[bool3,:] , y2mod.loc[bool3], cv=cv,
                            scoring=['r2','neg_mean_squared_error'],return_estimator=True)
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.median(np.sqrt(-scores['test_neg_mean_squared_error'])))
for i in range(len(y_var_da)):
    k=[]
    for j in range(5):
#        print(scores_collect[i][1]['estimator'][j].feature_importances_)
        if j==0:
            k = scores_collect[i][1]['estimator'][j].feature_importances_
        else:
            k=np.vstack((k,scores_collect[i][1]['estimator'][j].feature_importances_))
    #k2=np.vstack((k,dict_features_dp['IV_DP']))
    #k2=k2.T
    plt.figure()
    sns.heatmap(k,annot=True,xticklabels=X_da.columns);plt.tight_layout()
    plt.figure();ax=sns.barplot(data=k,ci='sd')
    ticks=ax.get_xticklabels()
    x_labels = list(ticks)
    plt.xticks(ticks=np.arange(X_da.columns.shape[0]),labels=X_da.columns)
    plt.title(y_var_da[i])
    plt.ylabel('Importância de cada variável')


#%% desparafinacao
dict_features_dp={'RendDP':	['IR_R','T_desaro','sqrt_T_desaro','square_T_desaro','square_RSO_desaro','square_T_despar','square_RSO_despar'],
'd_DP':	['IV_R','IR_R','T_desaro','RSO_despar','log_T_desaro','sqrt_T_desaro','sqrt_RSO_desaro','square_T_desaro','square_RSO_desaro','inter_T_desaro_RSO_desaro','inter_T_desaro_RSO_despar','inter_RSO_desaro_RSO_despar'],
'IR_DP':	['IV_R','IR_R','T_desaro','log_T_desaro','sqrt_T_desaro','sqrt_RSO_desaro','square_T_desaro','inter_T_desaro_RSO_desaro','inter_T_desaro_RSO_despar','inter_RSO_desaro_RSO_despar','inter_T_logRSO'],
'IV_DP':	['IR_R','T_desaro','RSO_despar','log_T_desaro','log_RSO_desaro','sqrt_T_desaro','square_T_desaro','square_RSO_desaro','inter_RSO_desaro_RSO_despar']}





list_exp=['log','sqrt','square','inter']

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
X_dp2['inter_T_logRSO']=X_dp2['T_desaro']*X_dp2['log_RSO_desaro']            

cv = KFold(n_splits=5, shuffle=True, random_state=358)

scores_collect=[]
SC=StandardScaler()
for i in y_var_dp:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(clf,SC.fit_transform(X_dp2.loc[bool3,dict_features_dp[i]]) , y2mod.loc[bool3], 
                            cv=cv,scoring=['r2','neg_mean_squared_error'],return_estimator=True)
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.mean(np.sqrt(-scores['test_neg_mean_squared_error'])))
   
k=[]
for i in range(5):
    print(scores_collect[-1][1]['estimator'][i].coef_,scores_collect[-1][1]['estimator'][i].intercept_)
    if i==0:
        k = scores_collect[-1][1]['estimator'][i].coef_
    else:
        k=np.vstack((k,scores_collect[-1][1]['estimator'][i].coef_))
k2=np.vstack((k,dict_features_dp['IV_DP']))
k2=k2.T
plt.figure()
sns.heatmap(k,annot=True)#,xticklabels=dict_deatures_dp['IV_DP'])
#%%
scores_collect=[]
for i in y_var_dp:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(clf,SC.fit_transform(X_dp.loc[bool3,:]) , y2mod.loc[bool3], 
                            cv=cv,scoring=['r2','neg_mean_squared_error'],return_estimator=True)
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.mean(np.sqrt(-scores['test_neg_mean_squared_error'])))
k=[]
for i in range(5):
    print(scores_collect[-1][1]['estimator'][i].coef_,scores_collect[-1][1]['estimator'][i].intercept_)
    if i==0:
        k = scores_collect[-1][1]['estimator'][i].coef_
    else:
        k=np.vstack((k,scores_collect[-1][1]['estimator'][i].coef_))
k2=np.vstack((k,X_dp.columns))
k2=k2.T
plt.figure()
sns.heatmap(k,annot=True,xticklabels=X_dp.columns);plt.tight_layout()

#%%
scores_collect=[]
x_var_dp=[ 'IV_R', 'd_R',  'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']

X_dp=dados_tot.loc[:,x_var_dp]

for i in y_var_dp:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(GBR,SC.fit_transform(X_dp.loc[bool3,:]) , y2mod.loc[bool3], 
                            cv=cv,scoring=['r2','neg_mean_squared_error'],return_estimator=True)
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.mean(np.sqrt(-scores['test_neg_mean_squared_error'])))

k=[]
for i in range(5):
    print(scores_collect[-1][1]['estimator'][i].feature_importances_)
    if i==0:
        k = scores_collect[-1][1]['estimator'][i].feature_importances_
    else:
        k=np.vstack((k,scores_collect[-1][1]['estimator'][i].feature_importances_))
#k2=np.vstack((k,dict_features_dp['IV_DP']))
#k2=k2.T
plt.figure()
sns.heatmap(k,annot=True,xticklabels=X_dp.columns);plt.tight_layout()
plt.figure();ax=sns.barplot(data=k,ci='sd')
ticks=ax.get_xticklabels()
x_labels = list(ticks)
plt.xticks(ticks=np.arange(X_dp.columns.shape[0]),labels=X_dp.columns)

for i in range(len(y_var_dp)):
    k=[]
    for j in range(5):
#        print(scores_collect[i][1]['estimator'][j].feature_importances_)
        if j==0:
            k = scores_collect[i][1]['estimator'][j].feature_importances_
        else:
            k=np.vstack((k,scores_collect[i][1]['estimator'][j].feature_importances_))
 #   plt.figure()
#    sns.heatmap(k,annot=True,xticklabels=X_dp.columns);plt.tight_layout()
    plt.figure();ax=sns.barplot(data=k,ci='sd')
    ticks=ax.get_xticklabels()
    x_labels = list(ticks)
    plt.xticks(ticks=np.arange(X_dp.columns.shape[0]),labels=X_dp.columns)
    plt.title(y_var_dp[i])
    plt.ylabel('Importância de cada variável')
    plt.tight_layout()
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\Imp_variaveis' +'\\var_imp_'+y_var_dp[i]+'.png',dpi=200)
plt.figure();sns.lmplot(data=dados_tot,x='d_R',y='d_DP',hue='Tipo',fit_reg=False)

#%% modelo residual
y2mod=dados_tot.loc[:,'d_DP']
bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                       np.logical_not(np.sum(np.isnan(X_dp2),axis=1)>=1).values)
bool3=np.logical_and(bool1,bool2)

clf.fit(dados_tot.loc[bool3,'d_R'].values.reshape(-1, 1),dados_tot.loc[bool3,'d_DP'])

res_d_dp=dados_tot.loc[bool3,'d_DP']-clf.predict(dados_tot.loc[bool3,'d_R'].values.reshape(-1, 1))
plt.figure();plt.plot(res_d_dp,dados_tot.loc[bool3,'d_R'],'o')
plt.xlabel('residuo do modelo linear')
plt.ylabel('densidade do rafinado')
plt.figure();plt.plot(res_d_dp,dados_tot.loc[bool3,'IV_R'],'o')
plt.xlabel('residuo do modelo linear')
plt.ylabel('densidade do rafinado')

clf.fit(dados_tot.loc[bool3,'IR_R'].values.reshape(-1, 1),dados_tot.loc[bool3,'IR_DP'])
res_IR_dp=dados_tot.loc[bool3,'IR_DP']-clf.predict(dados_tot.loc[bool3,'IR_R'].values.reshape(-1, 1))
plt.figure();plt.plot(res_IR_dp,dados_tot.loc[bool3,'IR_R'],'o')
plt.xlabel('residuo do modelo linear')
plt.ylabel('indice de refracao do rafinado')
plt.figure();plt.plot(res_IR_dp,dados_tot.loc[bool3,'IV_R'],'o')
plt.xlabel('residuo do modelo linear')
plt.ylabel('indice de refracao do rafinado')

res_d_dp.std()

X_res=dados_tot.loc[bool3,x_var_dp]
bool1=np.logical_and(res_d_dp<res_d_dp.mean()+3*res_d_dp.std(),res_d_dp>res_d_dp.mean()-3*res_d_dp.std())
bool2=np.logical_and(np.logical_not(np.isnan(res_d_dp)).values,
                       np.logical_not(np.sum(np.isnan(X_res),axis=1)>=1).values)
bool3=np.logical_and(bool1,bool2)

scores = cross_validate(GBR,X_res.loc[bool3,:] , res_d_dp.loc[bool3], 
                            cv=cv,scoring=['r2','neg_mean_squared_error'],return_estimator=True)
print(i,np.mean(scores['test_r2']),np.mean(np.sqrt(-scores['test_neg_mean_squared_error'])))

k=[]
for i in range(5):
    print(scores['estimator'][i].feature_importances_)
    if i==0:
        k = scores['estimator'][i].feature_importances_
    else:
        k=np.vstack((k,scores['estimator'][i].feature_importances_))
plt.figure()
sns.heatmap(k,annot=True,xticklabels=X_dp.columns);plt.tight_layout()
plt.figure();ax=sns.barplot(data=k,ci='sd')
ticks=ax.get_xticklabels()
x_labels = list(ticks)
plt.xticks(ticks=np.arange(X_dp.columns.shape[0]),labels=X_dp.columns)
plt.title('residuo')