# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:09:49 2021

@author: Felipo Soares
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path_dados=r'E:\coppe\SICOL_antigo\Novos Treinamentos_2017'
path_fig=r'E:\coppe\SICOL_novo\figuras'
dados_da=pd.read_excel(path_dados+'\\dadosredes.xlsx',sheet_name='desaromatização',
                       header=1)
x_var_da=['T', 'RSO']

y_var_da=['Visc_60R',  'Visc100_R', 'IV_R', 'd_R', 'IR_R', 'RendR']

print(dados_da.corr().loc[x_var_da,y_var_da])
print(dados_da.corr().loc[y_var_da,y_var_da])
tipos=dados_da["Unnamed: 1"].unique()
for i in tipos:
    print(dados_da.loc[dados_da["Unnamed: 1"]==i,:].corr().loc[x_var_da,y_var_da].iloc[[0,1],:])
    print(dados_da.loc[dados_da["Unnamed: 1"]==i,:].corr().loc[y_var_da,y_var_da])
    CC=dados_da.loc[dados_da["Unnamed: 1"]==i,:].corr()
    plt.figure()
    sns.heatmap(CC.loc[x_var_da,y_var_da],center=0,annot=True)
    plt.title(i)
    plt.figure()
    sns.heatmap(CC.loc[y_var_da,y_var_da],center=0,annot=True)
    plt.title(i)
print(i)
    
    
#for i in x_var:
 #   for j in y_var:
#        g=sns.lmplot(x=i, y=j, hue="Unnamed: 1", data=dados_da)
 #       plt.savefig(path_fig+'\\desaromatizacao_'+i+'_'+j+'.png',dpi=200)
  #      plt.close()
        
dados_dp=pd.read_excel(path_dados+'\\dadosredes.xlsx',sheet_name='desparafinação',
                       header=1)

x_var=[ 'Visc_40C', 'Visc60_C',  'Visc100_C',
       'IV_C', 'd_C', 'IR_C', 'T', 'RSO', 'T.1', 'RSO.1', 'Lav.']

y_var=[ 'Visc_60R',  'Visc100_R', 'IV_R', 'd_R', 'IR_R',
       'RendR', 'RendDP', 'd_DP', 'IR_DP', 'Visc_40DP', 'PL_DP', 'Visc100_DP',
       'IV_DP']



for i in x_var:
    try:
        dados_dp.loc[:,i]=pd.to_numeric(dados_dp.loc[:,i])
    except:
        for j in range(dados_dp.shape[0]):
            try:
                dados_dp.loc[j,i]=pd.to_numeric(dados_dp.loc[j,i])
            except:
                rat=dados_dp.loc[j,i]
                rat=rat.split(':')[0]
                rat=rat.replace(',','.')
                dados_dp.loc[j,i]=pd.to_numeric(rat)
        dados_dp.loc[:,i]=pd.to_numeric(dados_dp.loc[:,i])
        
for i in y_var:
    try:
        dados_dp.loc[:,i]=pd.to_numeric(dados_dp.loc[:,i])
    except:
        for j in range(dados_dp.shape[0]):
            try:
                dados_dp.loc[j,i]=pd.to_numeric(dados_dp.loc[j,i])
            except:
                rat=dados_dp.loc[j,i]
                rat=rat.split(':')[0]
                rat=rat.split('(')[0]
                rat=rat.replace(',','.')
                dados_dp.loc[j,i]=pd.to_numeric(rat)
        dados_dp.loc[:,i]=pd.to_numeric(dados_dp.loc[:,i])
            
tipos=dados_dp["Unnamed: 1"].unique()
for i in tipos:
    CC=dados_dp.loc[dados_dp["Unnamed: 1"]==i,:].corr()
    plt.figure()
    sns.heatmap(CC.loc[x_var,y_var2],center=0,annot=True)
    plt.title(i)
    plt.figure()
    sns.heatmap(CC.loc[y_var2,y_var2],center=0,annot=True)
    plt.title(i)



#sns.diverging_palette(240, 10, n=9)
plt.figure()
CC=dados_dp.corr().loc[x_var,y_var]
sns.heatmap(CC,center=0,annot=True)
plt.figure()
CC=dados_dp.corr().loc[y_var,y_var]
sns.heatmap(CC,center=0,annot=True)
plt.figure()
CC=dados_dp.corr().loc[x_var,x_var]
sns.heatmap(CC,center=0,annot=True)

#for i in x_var:
#    for j in y_var:
#        g=sns.lmplot(x=i, y=j, hue="Unnamed: 1", ci=50, n_boot=50,
#                     data=dados_dp,fit_reg=False)
##        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
#        plt.savefig(path_fig+'\\desparafinacao_'+i+'_'+j+'.png',dpi=200)
#        plt.close()

import statsmodels.api as sm
X=dados_da[x_var_da]
X_dum=pd.get_dummies(dados_da['Unnamed: 1'])
X=pd.concat((X,X_dum),axis=1)
X2=(X-X.median())/X.std()
X2 = sm.add_constant(X2)
y=dados_da.loc[:,y_var_da]
mod = sm.OLS(y.loc[:,'d_R'],X)
res=mod.fit();print(res.summary())



x_var_dp=[ 'IV_C', 'd_C', 'IR_C', 'T', 'RSO', 'T.1', 'RSO.1', 'Lav.']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T', 'RSO', 'T.1', 'RSO.1', 'Lav.']

y_var=[ 'Visc_60R',  'Visc100_R', 'IV_R', 'd_R', 'IR_R',
       'RendR', 'RendDP', 'd_DP', 'IR_DP', 'Visc_40DP', 'PL_DP', 'Visc100_DP',
       'IV_DP']

y_var2=['IV_R', 'd_R', 'IR_R','RendR', 'RendDP', 'd_DP', 'IR_DP',  'PL_DP', 
       'IV_DP']


X=dados_dp.loc[:,x_var_dp]
X = sm.add_constant(X)
y=dados_dp.loc[:,y_var]

mod = sm.OLS(y.iloc[:,12],X, missing='drop')
res=mod.fit();print(res.summary())

for i in y_var2:
   mod = sm.OLS(y.loc[:,i],X, missing='drop')
   res=mod.fit();print(res.summary())
 
#%% visualizing the increase of IV/IR/d
   
dados_dp2=dados_dp.copy()
dados_dp2=dados_dp2.loc[dados_dp['Unnamed: 1']!='SPM',:]
dados_dp2=dados_dp2.loc[dados_dp['Unnamed: 1']!='SPB',:]
dados_dp2=dados_dp2.loc[dados_dp2['IV_R']<200,:]

dados_dp2['diff_IV_R']=dados_dp2['IV_R']-dados_dp2['IV_C']
dados_dp2['diff_IR_R']=dados_dp2['IR_R']-dados_dp2['IR_C']
dados_dp2['diff_d_R']=dados_dp2['d_R']-dados_dp2['d_C']
dados_dp2['diff_IV_DP']=dados_dp2['IV_DP']-dados_dp2['IV_C']
dados_dp2['diff_IR_DP']=dados_dp2['IR_DP']-dados_dp2['IR_C']
dados_dp2['diff_d_DP']=dados_dp2['d_DP']-dados_dp2['d_C']

y_var_diff=['diff_IV_R','diff_IR_R','diff_d_R','diff_IV_DP','diff_IR_DP','diff_d_DP']

plt.figure()
CC=dados_dp2.corr().loc[x_var_dp,y_var_diff]
sns.heatmap(CC,center=0,annot=True)
plt.figure()
CC=dados_dp2.corr().loc[y_var_diff,y_var_diff]
sns.heatmap(CC,center=0,annot=True)
for i in y_var_diff:
    for j in y_var_dp:
        g=sns.lmplot(x=j, y=i, hue="Unnamed: 1", ci=50, n_boot=50,
                     data=dados_dp2,fit_reg=False)
#        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
        plt.savefig(path_fig+'\\differential_properties\\desparafinacao_'+i+'_'+j+'.png',dpi=200)
        plt.close()
    for j in x_var_dp:
        g=sns.lmplot(x=j, y=i, hue="Unnamed: 1", ci=50, n_boot=50,
                     data=dados_dp2,fit_reg=False)
#        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
        plt.savefig(path_fig+'\\differential_properties\\desparafinacao_'+i+'_'+j+'.png',dpi=200)
        plt.close()

g=g=sns.lmplot(x='d_C', y='diff_d_DP', hue="Unnamed: 1", ci=50, n_boot=50,
                     data=dados_dp2,fit_reg=False)
#dados_dp2.plot[x='diff_IV_R',y=IV_C']