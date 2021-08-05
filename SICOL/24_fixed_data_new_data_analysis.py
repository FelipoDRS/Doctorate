# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:01:49 2021

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

dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao_20210713.csv')
dados_da=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao_20210713.csv')
dados_tot=pd.concat((dados_da,dados_dp))

dados_tot.loc[dados_tot['IV_R']<30,'IV_R']=np.nan
dados_tot.loc[dados_tot['IV_R']>200,'IV_R']=np.nan
dados_tot.loc[dados_tot['T_despar']>-1,'T_despar']=np.nan
dados_tot.loc[dados_tot['RendDP']<20,'RendDP']=np.nan
dados_tot.loc[dados_tot['Lav_despar']<1,'Lav_despar']=np.nan

x_var_da=['T_desaro', 'RSO_desaro','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'Porcentagem','IV_C', 'd_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]

var_tot=[ 'IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro',
       'T_despar', 'RSO_despar', 'IV_R', 'd_R', 'IR_R', 'RendR', 'RendDP', 'd_DP', 'IR_DP',
        'IV_DP',  'Porcentagem',
        'Lav_despar']
    
path_images=r'E:\coppe\SICOL_novo\figuras\Dados_22_07'
y_var_dp=['IV_R', 'd_R', 'IR_R','RendR', 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']
tipos=['NP','NM','NL']
for i in tipos:
    CC=dados_tot.loc[dados_tot['Tipo']==i,:].corr()
    plt.figure()
    sns.heatmap(CC.loc[x_var_dp,y_var_dp],annot=True,center=0)
    plt.title(i)
    plt.tight_layout()
    plt.savefig(path_images+'//CC_X_vs_Y'+i+'.png',dpi=200)
    plt.close()
for i in tipos:
    CC=dados_tot.loc[dados_tot['Tipo']==i,:].corr()
    plt.figure()
    sns.heatmap(CC.loc[y_var_dp,y_var_dp],annot=True,center=0)
    plt.title(i)
    plt.tight_layout()
    plt.savefig(path_images+'//CC_Y_vs_Y'+i+'.png',dpi=200)
    plt.close()
for i in tipos:
    CC=dados_tot.loc[dados_tot['Tipo']==i,:].corr()
    plt.figure()
    sns.heatmap(CC.loc[x_var_dp,x_var_dp],annot=True,center=0)
    plt.title(i)
    plt.tight_layout()
    plt.savefig(path_images+'//CC_X_vs_X'+i+'.png',dpi=200)
    plt.close()
plt.close()
for i in range(len(var_tot)):
    for j in range(i+1,len(var_tot)):
        plt.figure()
        sns.lmplot( x=var_tot[i], y=var_tot[j],hue='Tipo', data=dados_tot, 
                    x_ci='ci', scatter=True, fit_reg=False)
        plt.savefig(path_images+'//lm_plot_'+var_tot[i]+'_vs_'+var_tot[j]+'.png',dpi=200)
        plt.close()
        
CC=dados_tot.corr()
plt.figure()
sns.heatmap(CC.loc[x_var_dp,y_var_dp],annot=True,center=0)
plt.title(i)
plt.tight_layout()