# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:42:05 2021

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
x_var=['d_C', 'IV_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=['d_C', 'IV_R', 'd_R', 'IR_R', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]


y_var_dp=[ 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']
y_var=[ 'IV_R', 'd_R', 'IR_R', 'RendR', 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']


tipos=['NL','NM','NP']
for i in tipos:
    plt.figure()
    CC=dados_tot.loc[dados_tot['Tipo']==i,:].corr().loc[x_var,y_var]
    sns.heatmap(CC,center=0,annot=True)
    plt.title(i)
    plt.tight_layout()
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\CC_X_y_'+i+'.png',dpi=200)
    plt.figure()
    CC=dados_tot.loc[dados_tot['Tipo']==i,:].corr().loc[y_var,y_var]
    sns.heatmap(CC,center=0,annot=True)
    plt.title(i)
    plt.tight_layout()
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\CC_y_y_'+i+'.png',dpi=200)

diff_dc=dados_da['d_C']-dados_da['d_R']
plt.figure();
for i in np.unique(dados_da['Tipo']):
    boll1=dados_da['Tipo']==i
    plt.plot(diff_dc.index[boll1],diff_dc[boll1],'o',label=i)
plt.xlabel('indice')
plt.ylabel('Redução da densidade na desaromatização')
plt.legend()
plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\perda_dens.png',dpi=200)

plt.figure()
g=sns.lmplot(x='d_R', y='IR_R', hue="Tipo", ci=50, n_boot=50,
     data=dados_dp,fit_reg=False)
plt.tight_layout()
#        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\DAO_outlier.png',dpi=200)
plt.close()
path_fig=r'E:\coppe\SICOL_novo\figuras\Dados_0601'
for i in x_var:
    for j in y_var:
        plt.figure()
        g=sns.lmplot(x=i, y=j, hue="Tipo", ci=50, n_boot=50,
                     data=dados_tot,fit_reg=False)
#        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
        plt.tight_layout()
        plt.savefig(path_fig+'\\despara_'+i+'_'+j+'.png',dpi=200)
        plt.close()

        #%%
    
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
yhat_collect=[]
dict_error={'IV_R':	2,
'd_R':0.0005,
'IR_R':	0.0005,
'RendR':	0,'IV_DP':	2,
'd_DP':0.0005,
'IR_DP':	0.0005,
'RendDP':	0}

for i in y_var_da:
    y2mod=dados_tot.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_da2),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    X2mod=X_da2.loc[bool3,dict_features[i]].values
    y2mod=y2mod.loc[bool3].values
    yhat_collect=np.zeros(y2mod.shape[0])

    #scores = cross_validate(clf,X_da2.loc[bool3,dict_features[i]] , , cv=cv,scoring=['r2','neg_mean_squared_error'])
    for train_index, test_index in cv.split(X2mod):
        X_train, X_test = X2mod[train_index,:], X2mod[test_index,:]
        y_train, y_test = y2mod[train_index], y2mod[test_index]
        clf.fit(X_train,y_train)
        yhat=clf.predict(X_test)
        yhat_collect[test_index]=yhat
    #    print(r2_score(y_test,yhat),np.sqrt(np.mean((y_test-yhat)**2)))
    print(i,r2_score(y2mod,yhat_collect),np.sqrt(np.mean((yhat_collect-y2mod)**2)))
        
    plt.figure(figsize=(3,1.8));plt.plot(y2mod,yhat_collect,'o')
    plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
    plt.xlabel('Valores reais')
    plt.ylabel('Valores preditos')
    plt.title(i)
    plt.tight_layout()
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hatvsreal_'+i+'.png',dpi=200)

    res=yhat_collect-y2mod
    plt.figure(figsize=(3,1.8));plt.hist(res,bins=20)
    plt.title(i)
    plt.xlabel('Resíduo')
    plt.ylabel('Contagem')
    print(i,np.mean(np.abs(res)<=dict_error[i]))
    plt.tight_layout()
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hist_'+i+'.png',dpi=200)
    
#    scores_collect.append([i,scores])
 #   print(i,np.mean(scores['test_r2']),np.median(np.sqrt(-scores['test_neg_mean_squared_error'])))
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
    X2mod=X_dp2.loc[bool3,dict_features_dp[i]].values
    y2mod=y2mod.loc[bool3].values
    yhat_collect=np.zeros(y2mod.shape[0])

    #scores = cross_validate(clf,X_da2.loc[bool3,dict_features[i]] , , cv=cv,scoring=['r2','neg_mean_squared_error'])
    for train_index, test_index in cv.split(X2mod):
        X_train, X_test = X2mod[train_index,:], X2mod[test_index,:]
        y_train, y_test = y2mod[train_index], y2mod[test_index]
        clf.fit(X_train,y_train)
        yhat=clf.predict(X_test)
        yhat_collect[test_index]=yhat
    #    print(r2_score(y_test,yhat),np.sqrt(np.mean((y_test-yhat)**2)))
    print(i,r2_score(y2mod,yhat_collect),np.sqrt(np.mean((yhat_collect-y2mod)**2)))
        
    plt.figure(figsize=(3,1.8));plt.plot(y2mod,yhat_collect,'o')
    plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
    plt.xlabel('Valores reais')
    plt.ylabel('Valores preditos')
    plt.title(i)
    plt.tight_layout()
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hatvsreal_'+i+'.png',dpi=200)

    res=yhat_collect-y2mod
    plt.figure(figsize=(3,1.8));plt.hist(res,bins=20)
    plt.title(i)
    plt.xlabel('Resíduo')
    plt.ylabel('Contagem')
    print(i,np.mean(np.abs(res)<=dict_error[i]))
    plt.tight_layout()
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hist_'+i+'.png',dpi=200)
    
    
    
#%%
    
from sklearn.ensemble import GradientBoostingRegressor
GBR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=4, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=100, tol=0.000001, )


X_dp=dados_tot.loc[:,x_var_dp]
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
print(r2_score(y2mod,yhat_collect),np.sqrt(np.mean((y2mod-yhat_collect)**2)))

plt.figure(figsize=(3,1.8));plt.plot(y2mod,yhat_collect,'o')
plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
plt.xlabel('Valores reais')
plt.ylabel('Valores preditos')
plt.title('d_DP modelo misto')
plt.tight_layout()
plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hatvsreal_lindR.png',dpi=200)
res=yhat_collect-y2mod
plt.figure(figsize=(3,1.8));plt.hist(res,bins=20)
plt.title('d_DP modelo misto')
plt.xlabel('Resíduo')
plt.ylabel('Contagem')
print(i,np.mean(np.abs(res)<=0.0005))
plt.tight_layout()
plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hist_lin_dr.png',dpi=200)

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
plt.figure();plt.plot(y2mod,yhat_collect,'o')
plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
plt.xlabel('IR_DP real')
plt.ylabel('IR_DP predito')
print(r2_score(y2mod,yhat_collect),np.sqrt(np.mean((y2mod-yhat_collect)**2)))

plt.figure(figsize=(3,1.8));plt.plot(y2mod,yhat_collect,'o')
plt.plot([y2mod.min(),y2mod.max()],[y2mod.min(),y2mod.max()],'r--')
plt.xlabel('Valores reais')
plt.ylabel('Valores preditos')
plt.title('IR_DP modelo misto')
plt.tight_layout()
plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hatvsreal_lindR.png',dpi=200)
res=yhat_collect-y2mod
plt.figure(figsize=(3,1.8));plt.hist(res,bins=20)
plt.title('IR_DP modelo misto')
plt.xlabel('Resíduo')
plt.ylabel('Contagem')
print(i,np.mean(np.abs(res)<=0.0005))
plt.tight_layout()
plt.savefig(r'E:\coppe\SICOL_novo\figuras\Dados_0601\hist_lin_dr.png',dpi=200)
#%%

from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')

ax.scatter3D(dados_da['T_desaro'], dados_da['RSO_desaro'],dados_da['d_R'], cmap='Greens');

ax = plt.axes(projection='3d')

ax.scatter3D(dados_da['T_desaro'], dados_da['RSO_desaro'],dados_da['RendR'], cmap='Greens');
