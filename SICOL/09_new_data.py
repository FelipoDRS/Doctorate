# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:56:48 2021

@author: Felipo Soares
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path_data=r'E:\coppe\SICOL_novo\dados'
sicol=pd.ExcelFile(path_data+'\\CondiçõesExploratórias_Kuwait.xls')
sicol.sheet_names

SPB=sicol.parse('SPB')
dados_dp=pd.read_csv(path_data+'\\dados_desparafinacao_20210526.csv')

'Unnamed: 0', 'ORIGEM', 'Unnamed: 1', 'Visc_40C', 'Visc60_C',
       'Visc80_C', 'Visc100_C', 'IV_C', 'd_C', 'IR_C', 'rend_raf', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar', 'Visc_40R',
       'Visc_60R', 'Visc_80R', 'Visc100_R', 'IV_R', 'd_R', 'IR_R', 'RendR',
       'RendDP', 'd_DP', 'IR_DP', 'Visc_40DP', 'PL_DP', 'Visc100_DP', 'IV_DP'
       
#dados_dp.iloc[101:110,:]
for i in range(12):       
    dados_dp=dados_dp.append(pd.Series(), ignore_index=True)

dados=SPB
dados_dp.loc[101,'ORIGEM']='Kuwait 2013'
dados_dp.loc[101,'Unnamed: 1']='SPB'
dados_dp.loc[101,'Visc_40C']=dados.iloc[9,8]
dados_dp.loc[101,'Visc60_C']=dados.iloc[10,8]
dados_dp.loc[101,'Visc100_C']=dados.iloc[11,8]
dados_dp.loc[101,'IV_C']=dados.iloc[12,8]
dados_dp.loc[101,'d_C']=dados.iloc[13,8]
dados_dp.loc[101,'IR_C']=dados.iloc[14,8]
dados_dp.loc[101,'T_desaro']=dados.iloc[5,9]
dados_dp.loc[101,'RSO_desaro']=float(dados.iloc[6,9].split(':')[0])
dados_dp.loc[101,'T_despar']=dados.iloc[18,8]
dados_dp.loc[101,'RSO_despar']=float(dados.iloc[19,8].split(':')[0])
dados_dp.loc[101,'Lav_despar']=float(dados.iloc[20,8].split(':')[0].replace(',','.'))
dados_dp.loc[101,'Visc_40R']=dados.iloc[9,9]
dados_dp.loc[101,'Visc_60R']=dados.iloc[10,9]
dados_dp.loc[101,'Visc100_R']=dados.iloc[11,9]
dados_dp.loc[101,'IV_R']=dados.iloc[12,9]
dados_dp.loc[101,'d_R']=dados.iloc[13,9]
dados_dp.loc[101,'IR_R']=dados.iloc[14,9]
dados_dp.loc[101,'RendR']=dados.iloc[15,9]
dados_dp.loc[101,'Visc_40DP']=dados.iloc[24,8]
dados_dp.loc[101,'Visc100_DP']=dados.iloc[25,8]
dados_dp.loc[101,'IV_DP']=dados.iloc[26,8]
dados_dp.loc[101,'d_DP']=dados.iloc[27,8]
dados_dp.loc[101,'IR_DP']=dados.iloc[28,8]
dados_dp.loc[101,'RendDP']=dados.iloc[30,8]
#%% NL
NL=sicol.parse('NL')
dados=NL
dados_dp.loc[102,'ORIGEM']='Kuwait 2013'
dados_dp.loc[102,'Unnamed: 1']='NL'
dados_dp.loc[102,'Visc60_C']=dados.iloc[9,19]
dados_dp.loc[102,'Visc100_C']=dados.iloc[10,19]
dados_dp.loc[102,'IV_C']=dados.iloc[11,19]
dados_dp.loc[102,'d_C']=dados.iloc[12,19]
dados_dp.loc[102,'IR_C']=dados.iloc[13,19]
dados_dp.loc[102,'T_desaro']=dados.iloc[5,20]
dados_dp.loc[102,'RSO_desaro']=float(dados.iloc[6,20].split(':')[0])
dados_dp.loc[102,'T_despar']=dados.iloc[18,19]
dados_dp.loc[102,'RSO_despar']=float(dados.iloc[19,19].split(':')[0])
dados_dp.loc[102,'Lav_despar']=float(dados.iloc[20,19].split(':')[0].replace(',','.'))
dados_dp.loc[102,'Visc_60R']=dados.iloc[9,20]
dados_dp.loc[102,'Visc100_R']=dados.iloc[10,20]
dados_dp.loc[102,'IV_R']=dados.iloc[11,20]
dados_dp.loc[102,'d_R']=dados.iloc[12,20]
dados_dp.loc[102,'IR_R']=dados.iloc[13,20]
dados_dp.loc[102,'RendR']=dados.iloc[14,20]
dados_dp.loc[102,'Visc_40DP']=dados.iloc[24,19]
dados_dp.loc[102,'Visc100_DP']=dados.iloc[25,19]
dados_dp.loc[102,'IV_DP']=dados.iloc[26,19]
dados_dp.loc[102,'d_DP']=dados.iloc[27,19]
dados_dp.loc[102,'IR_DP']=dados.iloc[28,19]
dados_dp.loc[102,'RendDP']=dados.iloc[30,19]

#%% NL
NM=sicol.parse('NM')
dados=NM
dados_dp.loc[103:107,'ORIGEM']='Kuwait 2013'
dados_dp.loc[103:107,'Unnamed: 1']='NM'
dados_dp.loc[103:107,'Visc60_C']=dados.iloc[9,[18,20,22,24,26]].values
dados_dp.loc[103:107,'Visc100_C']=dados.iloc[10,[18,20,22,24,26]].values
dados_dp.loc[103:107,'IV_C']=dados.iloc[11,[18,20,22,24,26]].values
dados_dp.loc[103:107,'d_C']=dados.iloc[12,[18,20,22,24,26]].values
dados_dp.loc[103:107,'IR_C']=dados.iloc[13,[18,20,22,24,26]].values
dados_dp.loc[103:107,'T_desaro']=dados.iloc[5,[19,21,23,25,27]]
dados_dp.loc[103:107,'RSO_desaro']=[5,5,7,5,6]
dados_dp.loc[103:107,'T_despar']=dados.iloc[17,[18,20,22,24,26]].values
dados_dp.loc[103:107,'RSO_despar']=4
dados_dp.loc[103:107,'Lav_despar']=2
dados_dp.loc[103:107,'Visc_60R']=dados.iloc[9,[19,21,23,25,27]].values
dados_dp.loc[103:107,'Visc100_R']=dados.iloc[10,[19,21,23,25,27]].values
dados_dp.loc[103:107,'IV_R']=dados.iloc[11,[19,21,23,25,27]].values
dados_dp.loc[103:107,'d_R']=dados.iloc[12,[19,21,23,25,27]].values
dados_dp.loc[103:107,'IR_R']=dados.iloc[13,[19,21,23,25,27]].values
dados_dp.loc[103:107,'RendR']=dados.iloc[14,[19,21,23,25,27]].values
dados_dp.loc[103:107,'Visc_40DP']=dados.iloc[23,[18,20,22,24,26]].values
dados_dp.loc[103:107,'Visc100_DP']=dados.iloc[24,[18,20,22,24,26]].values
dados_dp.loc[103:107,'IV_DP']=dados.iloc[25,[18,20,22,24,26]].values
dados_dp.loc[103:107,'d_DP']=dados.iloc[26,[18,20,22,24,26]].values
dados_dp.loc[103:107,'IR_DP']=dados.iloc[27,[18,20,22,24,26]].values
dados_dp.loc[103:107,'RendDP']=dados.iloc[29,[18,20,22,24,26]].values

#%% NP
NP=sicol.parse('NP')
dados=NP
dados_dp.loc[108:110,'ORIGEM']='Kuwait 2013'
dados_dp.loc[108:110,'Unnamed: 1']='NP'
dados_dp.loc[108:110,'Visc60_C']=dados.iloc[9,[22,24,26]].values
dados_dp.loc[108:110,'Visc100_C']=dados.iloc[10,[22,24,26]].values
dados_dp.loc[108:110,'IV_C']=dados.iloc[11,[22,24,26]].values
dados_dp.loc[108:110,'d_C']=dados.iloc[12,[22,24,26]].values
dados_dp.loc[108:110,'IR_C']=dados.iloc[13,[22,24,26]].values
dados_dp.loc[108:110,'T_desaro']=dados.iloc[5,[23,25,27]]
dados_dp.loc[108:110,'RSO_desaro']=[9,9,10]
dados_dp.loc[108:110,'T_despar']=dados.iloc[17,[22,24,26]].values
dados_dp.loc[108:110,'RSO_despar']=2.6
dados_dp.loc[108:110,'Lav_despar']=2
dados_dp.loc[108:110,'Visc_60R']=dados.iloc[9,[23,25,27]].values
dados_dp.loc[108:110,'Visc100_R']=dados.iloc[10,[23,25,27]].values
dados_dp.loc[108:110,'IV_R']=dados.iloc[11,[23,25,27]].values
dados_dp.loc[108:110,'d_R']=dados.iloc[12,[23,25,27]].values
dados_dp.loc[108:110,'IR_R']=dados.iloc[13,[23,25,27]].values
dados_dp.loc[108:110,'RendR']=dados.iloc[14,[23,25,27]].values
dados_dp.loc[108:110,'Visc_40DP']=dados.iloc[23,[22,24,26]].values
dados_dp.loc[108:110,'Visc100_DP']=dados.iloc[24,[22,24,26]].values
dados_dp.loc[108:110,'IV_DP']=dados.iloc[25,[22,24,26]].values
dados_dp.loc[108:110,'d_DP']=dados.iloc[26,[22,24,26]].values
dados_dp.loc[108:110,'IR_DP']=dados.iloc[27,[22,24,26]].values
dados_dp.loc[108:110,'RendDP']=dados.iloc[29,[22,24,26]].values

dados_dp.to_csv(path_data+r'\\dados_desparafinacao_20210526.csv',index=False)
#%%##%
dados_dp=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desparafinacao_20210526.csv')

dados_medanito=pd.read_excel(path_data+r'\\dados_medanito.xlsx')

def medanito2line(X,line,type_oil):
    dados_dp.loc[line,'ORIGEM']='Medanito'
    dados_dp.loc[line,'Unnamed: 1']=type_oil
    dados_dp.loc[line,'Visc60_C']=np.nan
    dados_dp.loc[line,'Visc100_C']=np.nan
    dados_dp.loc[line,'IV_C']=X[2,0]
    dados_dp.loc[line,'d_C']=X[0,0]
    dados_dp.loc[line,'IR_C']=X[1,0]
    dados_dp.loc[line,'T_desaro']=X[4,1]
    dados_dp.loc[line,'RSO_desaro']=X[5,1]
    dados_dp.loc[line,'T_despar']=X[4,2]
    dados_dp.loc[line,'RSO_despar']=X[5,2]
    dados_dp.loc[line,'Lav_despar']=X[6,2]
    dados_dp.loc[line,'Visc_60R']=np.nan
    dados_dp.loc[line,'Visc100_R']=np.nan
    dados_dp.loc[line,'IV_R']=X[2,1]
    dados_dp.loc[line,'d_R']=X[0,1]
    dados_dp.loc[line,'IR_R']=X[1,1]
    dados_dp.loc[line,'RendR']=X[3,1]
    dados_dp.loc[line,'Visc_40DP']=np.nan
    dados_dp.loc[line,'Visc100_DP']=np.nan
    dados_dp.loc[line,'IV_DP']=X[2,2]
    dados_dp.loc[line,'d_DP']=X[0,2]
    dados_dp.loc[line,'IR_DP']=X[1,2]
    dados_dp.loc[line,'RendDP']=X[3,2]

medanito2line(dados_medanito.iloc[0:7,1:4].values,111,'NL')
medanito2line(dados_medanito.iloc[0:7,7:10].values,112,'NM')
medanito2line(dados_medanito.iloc[9:16,1:4].values,113,'NP')
medanito2line(dados_medanito.iloc[9:16,7:10].values,114,'NL')
medanito2line(dados_medanito.iloc[19:26,1:4].values,115,'NM')
medanito2line(dados_medanito.iloc[19:26,7:10].values,116,'NP')
medanito2line(dados_medanito.iloc[29:36,1:4].values,117,'NL')
medanito2line(dados_medanito.iloc[29:36,[1,4,5]].values,118,'NL')
medanito2line(dados_medanito.iloc[29:36,[1,6,7]].values,119,'NL')
medanito2line(dados_medanito.iloc[39:46,1:4].values,120,'NM')
medanito2line(dados_medanito.iloc[39:46,[1,4,5]].values,121,'NM')
medanito2line(dados_medanito.iloc[39:46,[1,6,7]].values,122,'NM')
medanito2line(dados_medanito.iloc[48:55,1:4].values,123,'NP')
medanito2line(dados_medanito.iloc[48:55,[1,4,5]].values,124,'NP')
medanito2line(dados_medanito.iloc[48:55,[1,6,7]].values,125,'NP')

dados_dp.to_csv(path_data+r'\\dados_desparafinacao_20210526.csv',index=False)

#%%
for i in range(2):       
    dados_dp=dados_dp.append(pd.Series(), ignore_index=True)

sicol=pd.ExcelFile(path_data+'\\Basrah2.xls')
sicol.sheet_names

SPB=sicol.parse('SPB')
dados_dp=pd.read_csv(path_data+'\\dados_desparafinacao_20210526.csv')

NL=sicol.parse('NL')
dados=NL
dados_dp.loc[126,'ORIGEM']='Basrah'
dados_dp.loc[126,'Unnamed: 1']='NL'
dados_dp.loc[126,'Visc60_C']=dados.iloc[9,25]
dados_dp.loc[126,'Visc100_C']=dados.iloc[10,25]
dados_dp.loc[126,'IV_C']=dados.iloc[11,25]
dados_dp.loc[126,'d_C']=dados.iloc[12,25]
dados_dp.loc[126,'IR_C']=dados.iloc[13,25]
dados_dp.loc[126,'T_desaro']=dados.iloc[5,26]
dados_dp.loc[126,'RSO_desaro']=float(dados.iloc[6,26].split(':')[0])
dados_dp.loc[126,'T_despar']=dados.iloc[18,25]
dados_dp.loc[126,'RSO_despar']=float(dados.iloc[19,25].split(':')[0])
dados_dp.loc[126,'Lav_despar']=float(dados.iloc[20,25].split(':')[0].replace(',','.'))
dados_dp.loc[126,'Visc_60R']=dados.iloc[9,26]
dados_dp.loc[126,'Visc100_R']=dados.iloc[10,26]
dados_dp.loc[126,'IV_R']=dados.iloc[11,26]
dados_dp.loc[126,'d_R']=dados.iloc[12,26]
dados_dp.loc[126,'IR_R']=dados.iloc[13,26]
dados_dp.loc[126,'RendR']=dados.iloc[14,26]
dados_dp.loc[126,'Visc_40DP']=dados.iloc[24,25]
dados_dp.loc[126,'Visc100_DP']=dados.iloc[25,25]
dados_dp.loc[126,'IV_DP']=dados.iloc[26,25]
dados_dp.loc[126,'d_DP']=dados.iloc[27,25]
dados_dp.loc[126,'IR_DP']=dados.iloc[28,25]
dados_dp.loc[126,'RendDP']=dados.iloc[30,25]

#%%
NM=sicol.parse('NM')
dados=NM
dados_dp.loc[127,'ORIGEM']='Basrah'
dados_dp.loc[127,'Unnamed: 1']='NM'
dados_dp.loc[127,'Visc60_C']=dados.iloc[9,28]
dados_dp.loc[127,'Visc100_C']=dados.iloc[10,28]
dados_dp.loc[127,'IV_C']=dados.iloc[11,28]
dados_dp.loc[127,'d_C']=dados.iloc[12,28]
dados_dp.loc[127,'IR_C']=dados.iloc[13,28]
dados_dp.loc[127,'T_desaro']=dados.iloc[5,29]
dados_dp.loc[127,'RSO_desaro']=float(dados.iloc[6,29].split(':')[0])
dados_dp.loc[127,'T_despar']=dados.iloc[17,28]
dados_dp.loc[127,'RSO_despar']=float(dados.iloc[18,28].split(':')[0].replace(',','.'))
dados_dp.loc[127,'Lav_despar']=float(dados.iloc[19,28].split(':')[0].replace(',','.'))
dados_dp.loc[127,'Visc_60R']=dados.iloc[9,29]
dados_dp.loc[127,'Visc100_R']=dados.iloc[10,29]
dados_dp.loc[127,'IV_R']=dados.iloc[11,29]
dados_dp.loc[127,'d_R']=dados.iloc[12,29]
dados_dp.loc[127,'IR_R']=dados.iloc[13,29]
dados_dp.loc[127,'RendR']=dados.iloc[14,29]
dados_dp.loc[127,'Visc_40DP']=dados.iloc[23,28]
dados_dp.loc[127,'Visc100_DP']=dados.iloc[24,28]
dados_dp.loc[127,'IV_DP']=dados.iloc[25,28]
dados_dp.loc[127,'d_DP']=dados.iloc[26,28]
dados_dp.loc[127,'IR_DP']=dados.iloc[27,28]
dados_dp.loc[127,'RendDP']=dados.iloc[29,28]

dados_dp.to_csv(path_data+r'\\dados_desparafinacao_20210531.csv',index=False)
#%%

path_data=r'E:\coppe\SICOL_novo\dados'
sicol=pd.read_excel(path_data+'\\BS_AL_FURFURAL.xls')
sicol.sheet_names

dados_da=pd.read_csv(path_data+'\\dados_desaromatizacao.csv')

'Unnamed: 0', 'ORIGEM', 'Unnamed: 1', 'Visc_40C', 'Visc60_C',
       'Visc80_C', 'Visc100_C', 'IV_C', 'd_C', 'IR_C', 'rend_raf', 'T_desaro',
       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar', 'Visc_40R',
       'Visc_60R', 'Visc_80R', 'Visc100_R', 'IV_R', 'd_R', 'IR_R', 'RendR',
       'RendDP', 'd_DP', 'IR_DP', 'Visc_40DP', 'PL_DP', 'Visc100_DP', 'IV_DP'
       
#dados_dp.iloc[101:110,:]
for i in range(36):       
    dados_da=dados_da.append(pd.Series(), ignore_index=True)
bool_carga=sicol['Unnamed: 3']=='RAFINADO'
dados_da.loc[109:,'ORIGEM']=sicol.loc[bool_carga,'Unnamed: 4'].values
dados_da.loc[109:,'Unnamed: 1']=sicol.loc[bool_carga,'Unnamed: 1'].values
dados_da.loc[109:,'d_C']=sicol.loc[6,'Unnamed: 14']
dados_da.loc[109:,'IR_C']=sicol.loc[6,'Unnamed: 15']
dados_da.loc[109:,'T']=sicol.loc[bool_carga,'Unnamed: 5'].values
dados_da.loc[109:,'RSO']=sicol.loc[bool_carga,'Unnamed: 6'].values
dados_da.loc[109:,'Visc100_R']=sicol.loc[bool_carga,'Unnamed: 18'].values
dados_da.loc[109:,'d_R']=sicol.loc[bool_carga,'Unnamed: 14'].values
dados_da.loc[109:,'IR_R']=sicol.loc[bool_carga,'Unnamed: 15'].values
dados_da.loc[109:,'RendR']=sicol.loc[bool_carga,'Unnamed: 11'].values
dados_da=dados_da.rename({'Unnamed: 1':'Tipo'})
dados_da.columns=['Unnamed: 0', 'ORIGEM', 'Tipo', 'Visc_40C', 'Visc60_C',
       'Visc80_C', 'Visc100_C', 'IV_C', 'd_C', 'IR_C', 'T', 'RSO', 'T.1',
       'RSO.1', 'Lav.', 'Visc_40R', 'Visc_60R', 'Visc_80R', 'Visc100_R',
       'IV_R', 'd_R', 'IR_R', 'RendR', 'RendDP', 'd_DP', 'IR_DP', 'Visc_40DP',
       'PL_DP', 'Visc100_DP', 'IV_DP']
dados_da.to_csv(path_data+r'\\dados_desaromatizacao_20210601.csv',index=False)

