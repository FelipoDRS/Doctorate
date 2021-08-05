# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:43:58 2021

@author: Felipo Soares
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path_dados=r'E:\coppe\SICOL_antigo\Novos Treinamentos_2017'
path_fig=r'E:\coppe\SICOL_novo\figuras'

        
dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao_20210531.csv')

x_var_da=['T', 'RSO','d_C']

y_var_da=[ 'IV_R', 'd_R', 'IR_R', 'RendR']

x_var_dp=[ 'IV_C', 'd_C', 'IR_C', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar']
x_var_C=[ 'IR_C',   'IV_C', 'd_C']
x_var_op=[ 'T_desaro', 'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar',]
y_var_dp=['IV_R', 'd_R', 'IR_R','RendR', 'RendDP', 'd_DP', 'IR_DP', 'IV_DP']

dict_feature={'IV_R':['IV_C', 'd_C', 'IR_C', 'RSO_desaro'],
               'd_R':['IV_C', 'd_C', 'IR_C', 'RSO_desaro'], 
               'IR_R':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro'],
               'RendR':[ 'IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro'],
               'RendDP':['d_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'T_despar','RSO_despar'],
               'd_DP':['IV_C', 'd_C', 'IR_C',  'RSO_desaro','RSO_despar', ],
               'IR_DP':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IV_DP':['IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', ]}

dict_feature_lin={'IV_R':['const','logrso','IV_C', 'd_C', 'IR_C', 'RSO_desaro'],
               'd_R':['const','IV_C', 'd_C', 'IR_C', 'RSO_desaro'], 
               'IR_R':['const','IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro'],
               'RendR':[ 'const','IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro'],
               'RendDP':['const','d_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'T_despar','RSO_despar'],
               'd_DP':['const','IV_C', 'd_C', 'IR_C',  'RSO_desaro','RSO_despar', ],
               'IR_DP':['const','IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro','RSO_despar', 'Lav_despar'],
               'IV_DP':['const','IV_C', 'd_C', 'IR_C', 'T_desaro', 'RSO_desaro', 'RSO_despar', ]}



            
tipos=dados_dp["Unnamed: 1"].unique()
for i in ['NL','NP','NM']:
    CC=dados_dp.loc[dados_dp["Unnamed: 1"]==i,:].corr()
    plt.figure()
    sns.heatmap(CC.loc[x_var_dp,y_var_dp],center=0,annot=True)
    plt.title(i)
    plt.figure()
    sns.heatmap(CC.loc[y_var_dp,y_var_dp],center=0,annot=True)
    plt.title(i)

bool_novos=np.logical_or(np.logical_or(dados_dp['ORIGEM']=='Medanito',dados_dp['ORIGEM']=='Kuwait 2013'),dados_dp['ORIGEM']=='Basrah')
novos_dados=dados_dp.loc[bool_novos,:]
velhos_dados=dados_dp.loc[np.logical_not(bool_novos),:]



print((novos_dados.mean()-velhos_dados.mean())/velhos_dados.std())
for i in ['NL','NP','NM']:
    print('\n' +i+' \n')
    print((novos_dados.loc[novos_dados['Unnamed: 1']==i,:].mean()-velhos_dados.loc[velhos_dados['Unnamed: 1']==i,:].mean())
    /velhos_dados.loc[velhos_dados['Unnamed: 1']==i,:].std())
    

#sns.diverging_palette(240, 10, n=9)

plt.figure()
plt.plot(novos_dados['d_R'],novos_dados['d_DP'],'ro')
plt.plot(velhos_dados['d_R'],velhos_dados['d_DP'],'bo')

plt.figure()
plt.plot(novos_dados['IV_C'],novos_dados['IV_DP'],'ro')
plt.plot(velhos_dados['IV_C'],velhos_dados['IV_DP'],'bo')


#for i in x_var:
#    for j in y_var:
#        g=sns.lmplot(x=i, y=j, hue="Unnamed: 1", ci=50, n_boot=50,
#                     data=dados_dp,fit_reg=False)
##        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
#        plt.savefig(path_fig+'\\desparafinacao_'+i+'_'+j+'.png',dpi=200)
#        plt.close()


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
y=dados_dp.loc[:,y_var_dp]
X['logrso']=np.log(X['RSO_desaro'])
mod = sm.OLS(y.iloc[:,1],X, missing='drop')
res=mod.fit();print(res.summary())

for i in y_var_dp:
   mod = sm.OLS(y.loc[:,i],X, missing='drop')
   res=mod.fit();print(res.summary())
for i in y_var_dp:
   mod = sm.OLS(y.loc[:,i],X.loc[:,dict_feature_lin[i]], missing='drop')
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
#%%
x_var_da=['T', 'RSO']

y_var_da=['Visc_60R',  'Visc100_R', 'IV_R', 'd_R', 'IR_R', 'RendR']
path_fig=r'E:\coppe\SICOL_novo\figuras\Dados_0601'
for i in x_var_da:
    for j in y_var_da:
        g=sns.lmplot(x=i, y=j, hue="Tipo", ci=70, n_boot=150,
                     data=dados_da,fit_reg=True)
#        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
        plt.savefig(path_fig+'\\desaromatzacao_'+i+'_'+j+'.png',dpi=200)
        plt.close()

for i in range(len(y_var_da)):
    for j in range(i,len(y_var_da)):
        if i != j:
            g=sns.lmplot(x=y_var_da[i], y=y_var_da[j], hue="Tipo", ci=50, n_boot=50,
                         data=dados_da,fit_reg=False)
    #        plt.ylim([dados_dp.loc[:,j].min()*0.90,dados_dp.loc[:,j].max()*1.1])
            plt.savefig(path_fig+'\\desaromatzacao_'+y_var_da[i]+'_'+y_var_da[j]+'.png',dpi=200)
            plt.close()

sns