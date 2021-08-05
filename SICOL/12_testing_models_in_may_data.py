# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:26:09 2021

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
from sklearn.linear_model import LinearRegression, RidgeCV,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

path_dados=r'E:\coppe\SICOL_antigo\Novos Treinamentos_2017'
path_fig=r'E:\coppe\SICOL_novo\figuras'

        
dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao_20210531.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao_20210601.csv')

x_var_da=['T', 'RSO','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_C', 'd_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]
y_var_dp=['IV_R', 'd_R', 'IR_R','RendR', 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']


bool_novos=np.logical_or(np.logical_or(dados_dp['ORIGEM']=='Medanito',dados_dp['ORIGEM']=='Kuwait 2013'),dados_dp['ORIGEM']=='Basrah')
novos_dados=dados_dp.loc[bool_novos,:]
velhos_dados=dados_dp.loc[np.logical_not(bool_novos),:]



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

dict_features_dp={'IV_R':	['IV_C','T_desaro','T_despar','RSO_despar','log_T_desaro',
                           'sqrt_RSO_desaro','square_T_despar',
                           'inter_T_desaro_T_despar','inter_RSO_desaro_T_despar',
                           'inter_T_despar_RSO_despar'],
'd_R':	['IR_C','T_desaro','RSO_desaro','log_RSO_despar','sqrt_RSO_despar',
'square_T_desaro','inter_T_desaro_RSO_despar'],
'IR_R':	['IR_C','T_desaro','RSO_desaro','log_RSO_despar','sqrt_RSO_despar',
'square_T_desaro','inter_T_desaro_RSO_despar'],
'RendR':	['IV_C','IR_C','T_desaro','RSO_despar','sqrt_RSO_despar',
'square_T_desaro','square_RSO_desaro'],
'RendDP':	['IV_C','log_T_desaro','inter_T_desaro_RSO_despar',
'inter_RSO_desaro_RSO_despar','inter_T_despar_RSO_despar'],
'd_DP':	['IR_C','T_desaro','RSO_desaro','log_RSO_despar','sqrt_RSO_despar',
'square_T_desaro','inter_T_desaro_RSO_despar'],
'IR_DP':	['IV_C','IR_C','T_desaro','RSO_desaro','log_T_desaro',
'sqrt_T_desaro','sqrt_RSO_despar','square_RSO_despar'],
'IV_DP':	['IV_C','IR_C','log_RSO_despar','sqrt_RSO_desaro',
'inter_T_desaro_RSO_despar','inter_T_logRSO'],
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

X_dp_train=X_dp2.loc[np.logical_not(bool_novos),:]
X_dp_test=X_dp2.loc[bool_novos,:]
clf=LinearRegression()
scores_collect=[]
for i in y_var_dp:
    y2mod=dados_dp.loc[np.logical_not(bool_novos),i]
    y2test=dados_dp.loc[bool_novos,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_train),axis=1)>=1).values),
                        np.logical_not(np.sum(np.isinf(X_dp_train),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    bool4=np.logical_and(np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_test),axis=1)>=1).values),
                           np.logical_not(np.sum(np.isinf(X_dp_test),axis=1)>=1).values),
                        np.logical_not(np.sum(X_dp_test==0,axis=1)>=1).values)
    
                         
    clf.fit(X_dp_train.loc[bool3,dict_features_dp[i]] , y2mod.loc[bool3])
    yhat=clf.predict(X_dp_test.loc[bool4,dict_features_dp[i]])
    rmse=np.sqrt(np.mean((yhat-y2test.loc[bool4])**2))
    print(i,r2_score(y2test.loc[bool4],yhat),rmse)
    plt.figure()
    plt.plot(y2test.loc[bool4],yhat,'ro');plt.title(i)    
    plt.plot([y2test.loc[bool4].min(),y2test.loc[bool4].max()],[y2test.loc[bool4].min(),y2test.loc[bool4].max()],'b--')
    plt.xlabel(i + ' real')
    plt.xlabel(i + ' predito')
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\plot dados novos maio' + '\\realvspredito ' + i +'.png')
#%% dummy model
for i in y_var_dp:
    y2mod=dados_dp.loc[np.logical_not(bool_novos),i]
    y2test=dados_dp.loc[bool_novos,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_train),axis=1)>=1).values),
                        np.logical_not(np.sum(np.isinf(X_dp_train),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    bool4=np.logical_and(np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_test),axis=1)>=1).values),
                           np.logical_not(np.sum(np.isinf(X_dp_test),axis=1)>=1).values),
                        np.logical_not(np.sum(X_dp_test==0,axis=1)>=1).values),
                         
    clf.fit(X_dp_train.loc[bool3,dict_features_dp[i]] , y2mod.loc[bool3])
    yhat=np.ones(y2test.loc[bool4].shape[0])*y2mod.loc[bool3].mean()
    rmse=np.sqrt(np.mean((yhat-y2test.loc[bool4])**2))
    print(i,r2_score(y2test.loc[bool4],yhat),rmse)
    plt.figure()
    plt.plot(y2test.loc[bool4],yhat,'ro');plt.title(i)    
    plt.plot([y2test.loc[bool4].min(),y2test.loc[bool4].max()],[y2test.loc[bool4].min(),y2test.loc[bool4].max()],'b--')
    
#%% simple linear model    
for i in y_var_dp:
    y2mod=dados_dp.loc[np.logical_not(bool_novos),i]
    y2test=dados_dp.loc[bool_novos,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_train),axis=1)>=1).values),
                        np.logical_not(np.sum(np.isinf(X_dp_train),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    bool4=np.logical_and(np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_test),axis=1)>=1).values),
                           np.logical_not(np.sum(np.isinf(X_dp_test),axis=1)>=1).values),
                         np.logical_not(np.sum(X_dp_test==0,axis=1)>=1).values)
                        
    clf.fit(X_dp_train.loc[bool3,x_var_dp] , y2mod.loc[bool3])
    yhat=clf.predict(X_dp_test.loc[bool4,x_var_dp])
    rmse=np.sqrt(np.mean((yhat-y2test.loc[bool4])**2))
    print(i,r2_score(y2test.loc[bool4],yhat),rmse)
    plt.figure()
    plt.plot(y2test.loc[bool4],yhat,'ro');plt.title(i)    
    plt.plot([y2test.loc[bool4].min(),y2test.loc[bool4].max()],[y2test.loc[bool4].min(),y2test.loc[bool4].max()],'b--')
    plt.xlabel(i + ' real')
    plt.xlabel(i + ' predito')
    plt.savefig(r'E:\coppe\SICOL_novo\figuras\plot dados novos maio' + '\\realvspredito puro linear' + i +'.png')


#%%
from sklearn.model_selection import cross_val_score, cross_validate, KFold
cv = KFold(n_splits=5, shuffle=True, random_state=358)
SC=StandardScaler()
scores_collect=[]
for i in y_var_dp:
    y2mod=dados_dp.loc[:,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    scores = cross_validate(clf,SC.fit_transform(X_dp.loc[bool3,:]) , y2mod.loc[bool3], 
                            cv=cv,scoring=['r2','neg_mean_squared_error'],return_estimator=True)
    scores_collect.append([i,scores])
    print(i,np.mean(scores['test_r2']),np.mean(np.sqrt(-scores['test_neg_mean_squared_error'])))
    
#    print(np.sum(1-bool3))
for i in range(5):
    print(scores_collect[-1][1]['estimator'][i].coef_,scores_collect[-1][1]['estimator'][i].intercept_)
#%%
from sklearn.ensemble import GradientBoostingRegressor
GBR=GradientBoostingRegressor( learning_rate=0.03, n_estimators=200, subsample=0.9, 
                              min_samples_split=8, min_samples_leaf=3,
                              max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=200, tol=0.000001, )

scores_collect=[]
X_da_train=dados_dp.loc[np.logical_not(bool_novos),x_var_dp]
X_da_test=dados_dp.loc[bool_novos,x_var_dp]

for i in y_var_dp:
    y2mod=dados_dp.loc[np.logical_not(bool_novos),i]
    y2test=dados_dp.loc[bool_novos,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_train),axis=1)>=1).values),
                        np.logical_not(np.sum(np.isinf(X_dp_train),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    bool4=np.logical_and(np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                           np.logical_not(np.sum(np.isnan(X_dp_test),axis=1)>=1).values),
                           np.logical_not(np.sum(np.isinf(X_dp_test),axis=1)>=1).values),
    np.logical_not(np.sum(X_dp_test==0,axis=1)>=1))
                         
    GBR.fit(X_dp_train.loc[bool3,:] , y2mod.loc[bool3])
    yhat=GBR.predict(X_dp_test.loc[bool4,:])
    rmse=np.sqrt(np.mean((yhat-y2test.loc[bool4])**2))
    print(i,r2_score(y2test.loc[bool4],yhat),rmse)
    plt.figure()
    plt.plot(y2test.loc[bool4],yhat,'ro');plt.title(i)    
    plt.plot([y2test.loc[bool4].min(),y2test.loc[bool4].max()],[y2test.loc[bool4].min(),y2test.loc[bool4].max()],'b--')
#%%
bool_novos=dados_da.Tipo=='DEST. BS'
x_var_da=['T', 'RSO','d_C']
X_da_train=dados_da.loc[np.logical_not(bool_novos),x_var_da]
X_da_test=dados_da.loc[bool_novos,x_var_da]
from sklearn.model_selection import cross_val_score, cross_validate, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=358)
clf=LinearRegression()
y_var_da2=[ 'd_R', 'IR_R', 'RendR']
X_da_train['logrso']=np.log(X_da_train['RSO'])
X_da_test['logrso']=np.log(X_da_test['RSO'])

for i in y_var_da2:
    y2mod=dados_da.loc[np.logical_not(bool_novos),i]
    y2test=dados_da.loc[bool_novos,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_da_train),axis=1)>=1).values),
                        np.logical_not(np.sum(np.isinf(X_da_train),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    bool4=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                           np.logical_not(np.sum(np.isnan(X_da_test),axis=1)>=1).values),
                           np.logical_not(np.sum(np.isinf(X_da_test),axis=1)>=1).values)
                         
    clf.fit(X_da_train.loc[bool3,:] , y2mod.loc[bool3])
    yhat=clf.predict(X_da_test)
    rmse=np.sqrt(np.mean((yhat-y2test.loc[bool4])**2))
    print(i,r2_score(y2test.loc[bool4],yhat),rmse)
    plt.figure()
    plt.plot(y2test.loc[bool4],yhat,'ro');plt.title(i)    
    plt.plot([y2test.loc[bool4].min(),y2test.loc[bool4].max()],[y2test.loc[bool4].min(),y2test.loc[bool4].max()],'b--')
#%%
from sklearn.ensemble import GradientBoostingRegressor
GBR=GradientBoostingRegressor( learning_rate=0.04, n_estimators=200, subsample=0.9, 
                              min_samples_split=8, min_samples_leaf=3,
                              max_depth=6, min_impurity_decrease=0.0, min_impurity_split=None, 
                              random_state=324,  alpha=0.9, verbose=0, 
                              validation_fraction=0.05, n_iter_no_change=200, tol=0.000001, )

scores_collect=[]
X_da_train=dados_da.loc[np.logical_not(bool_novos),x_var_da]
X_da_test=dados_da.loc[bool_novos,x_var_da]
X_da_train['logrso']=np.log(X_da_train['RSO'])
X_da_test['logrso']=np.log(X_da_test['RSO'])

for i in y_var_da2:
    y2mod=dados_da.loc[np.logical_not(bool_novos),i]
    y2test=dados_da.loc[bool_novos,i]
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_da_train),axis=1)>=1).values),
                        np.logical_not(np.sum(np.isinf(X_da_train),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    bool4=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                           np.logical_not(np.sum(np.isnan(X_da_test),axis=1)>=1).values),
                           np.logical_not(np.sum(np.isinf(X_da_test),axis=1)>=1).values)
                         
    GBR.fit(X_da_train.loc[bool3,:] , y2mod.loc[bool3])
    yhat=GBR.predict(X_da_test.loc[bool4,:])
    rmse=np.sqrt(np.mean((yhat-y2test.loc[bool4])**2))
    print(i,r2_score(y2test.loc[bool4],yhat),rmse)
    plt.figure()
    plt.plot(y2test.loc[bool4],yhat,'ro');plt.title(i)    
    plt.plot([y2test.loc[bool4].min(),y2test.loc[bool4].max()],[y2test.loc[bool4].min(),y2test.loc[bool4].max()],'b--')
#%%
diff_dc=dados_da['d_C']-dados_da['d_R']
plt.figure();plt.plot(dados_da['T'],diff_dc,'o')
plt.plot(dados_da['RSO'],diff_dc,'o')

SC=StandardScaler()
y2mod=diff_dc.loc[np.logical_not(bool_novos)]
y2test=diff_dc.loc[bool_novos]
bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                       np.logical_not(np.sum(np.isnan(X_da_train),axis=1)>=1).values),
                    np.logical_not(np.sum(np.isinf(X_da_train),axis=1)>=1).values)
bool3=np.logical_and(bool1,bool2)
bool4=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                       np.logical_not(np.sum(np.isnan(X_da_test),axis=1)>=1).values),
                       np.logical_not(np.sum(np.isinf(X_da_test),axis=1)>=1).values)
                     
clf.fit(SC.fit_transform(X_da_train.loc[bool3,:]) , y2mod.loc[bool3])
yhat=clf.predict(SC.transform(X_da_test.loc[bool4,:]))
rmse=np.sqrt(np.mean((yhat-y2test.loc[bool4])**2))
print(i,r2_score(y2test.loc[bool4],yhat),rmse)
plt.figure()
plt.plot(y2test.loc[bool4],yhat,'ro');plt.title(i)    
plt.plot([y2test.loc[bool4].min(),y2test.loc[bool4].max()],[y2test.loc[bool4].min(),y2test.loc[bool4].max()],'b--')


#%% fazendo errado
X_da_train_norm=StandardScaler().fit_transform(X_da_train)
X_da_test_norm=StandardScaler().fit_transform(X_da_test)

for i in y_var_da2:
    y2mod=dados_da.loc[np.logical_not(bool_novos),i]
    y2test=dados_da.loc[bool_novos,i]
    
    y2mod_norm=StandardScaler().fit_transform(y2mod.values.reshape((-1,1))).reshape((-1,))
    y2test_norm=StandardScaler().fit_transform(y2test.values.reshape((-1,1))).reshape((-1,))
    
    bool1=np.logical_and(y2mod<y2mod.mean()+4*y2mod.std(),y2mod>y2mod.mean()-4*y2mod.std())
    bool2=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2mod)).values,
                           np.logical_not(np.sum(np.isnan(X_da_train),axis=1)>=1).values),
                        np.logical_not(np.sum(np.isinf(X_da_train),axis=1)>=1).values)
    bool3=np.logical_and(bool1,bool2)
    bool4=np.logical_and(np.logical_and(np.logical_not(np.isnan(y2test)).values,
                           np.logical_not(np.sum(np.isnan(X_da_test),axis=1)>=1).values),
                           np.logical_not(np.sum(np.isinf(X_da_test),axis=1)>=1).values)
                         
    GBR.fit(X_da_train_norm[bool3,:] , y2mod_norm[bool3])
    yhat=GBR.predict(X_da_test_norm[bool4,:])
    rmse=np.sqrt(np.mean((yhat-y2test_norm[bool4])**2))
    print(i,r2_score(y2test_norm[bool4],yhat),rmse)
    plt.figure()
    plt.plot(y2test_norm[bool4],yhat,'ro');plt.title(i)    
    plt.plot([y2test_norm[bool4].min(),y2test_norm[bool4].max()],[y2test_norm[bool4].min(),y2test_norm[bool4].max()],'b--')
