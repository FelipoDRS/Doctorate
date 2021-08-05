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
sicol=pd.ExcelFile(path_data+'\\SICOL.xls')
sicol.sheet_names

SPB=sicol.parse('SPB')
dados_dp=pd.read_csv(path_data+'\\dados_desparafinacao_20210709.csv')
dados_dp.drop('Unnamed: 0',inplace=True,axis=1)
def reorg_dados_carga(db,dados,ind_db,ind_dados,ident):
    '''
    db é a base de dados onde vão os dados
    dados é a tabela original
    ind_db é a linha
    ind_dados é a coluna do IV
    ident é a identidade do oleo
    '''
    row=ind_dados[0]
    col=ind_dados[1]
    
    db.loc[ind_db,'ORIGEM']=ident[0]
    db.loc[ind_db,'Tipo oleo']=np.nan
    db.loc[ind_db,'Tipo']=ident[1]
    db.loc[ind_db,'IV_C']=dados.iloc[row,col]
    db.loc[ind_db,'d_C']=dados.iloc[row+1,col]
    db.loc[ind_db,'IR_C']=dados.iloc[row+2,col]
    bool1= db.loc[ind_db,'IV_C']>50 and db.loc[ind_db,'IV_C']<100
    bool2= db.loc[ind_db,'IR_C']>1.4 and db.loc[ind_db,'IV_C']<1.5
    bool3= db.loc[ind_db,'d_C']>0.8 and db.loc[ind_db,'IV_C']<90
    if not bool1:
        print('erro no IV, carga')
    if not bool2:
        print('erro no IR, carga')
    if not bool3:
        print('erro no na densidade, carga')
    return db
    
def reorg_dados_da(db,dados,ind_db,ind_dados,ind_op):
    '''
    db é a base de dados onde vão os dados
    dados é a tabela original
    ind_db é a linha
    ind_dados é a coluna do IV
    ind_op é o indice dos dados operacionais
    '''
    row=ind_dados[0]
    col=ind_dados[1]
    row2=ind_op[0]
    col2=ind_op[1]
    
    db.loc[ind_db,'IV_R']=dados.iloc[row,col]
    db.loc[ind_db,'d_R']=dados.iloc[row+1,col]
    db.loc[ind_db,'IR_R']=dados.iloc[row+2,col]
    db.loc[ind_db,'RendR']=float(dados.iloc[row+3,col])

    db.loc[ind_db,'T_desaro']=float(dados.iloc[row2,col2])
    db.loc[ind_db,'RSO_desaro']=float(dados.iloc[row2+1,col2].split(':')[0].replace(',','.'))
    
    bool1= db.loc[ind_db,'IV_R']>50 and db.loc[ind_db,'IV_R']<100
    bool2= db.loc[ind_db,'IR_R']>1.4 and db.loc[ind_db,'IV_R']<1.5
    bool3= db.loc[ind_db,'d_R']>0.8 and db.loc[ind_db,'IV_R']<90
    bool4= db.loc[ind_db,'RendR']>10 and db.loc[ind_db,'RendR']<90
    bool5= db.loc[ind_db,'T_desaro']>50 and db.loc[ind_db,'T_desaro']<140
    bool6= db.loc[ind_db,'RSO_desaro']>0 and db.loc[ind_db,'RSO_desaro']<16


    if not bool1:
        print('erro no IV, rafinado')
    if not bool2:
        print('erro no IR, rafinado', [row,col],dados.iloc[row,col])
    if not bool3:
        print('erro na densidade, rafinado')
    if not bool4:
        print('erro no rendimento, rafinado')
    if not bool5:
        print('erro na temp, rafinado')
    if not bool6:
        print('erro no RSO, rafinado')

    return db

def reorg_dados_dp(db,dados,ind_db,ind_dados,ind_op):
    '''
    db é a base de dados onde vão os dados
    dados é a tabela original
    ind_db é a linha
    ind_dados é a coluna do IV
    ind_op é o indice dos dados operacionais
    '''
    row=ind_dados[0]
    col=ind_dados[1]
    row2=ind_op[0]
    col2=ind_op[1]
    
    db.loc[ind_db,'IV_DP']=dados.iloc[row,col]
    db.loc[ind_db,'d_DP']=dados.iloc[row+1,col]
    db.loc[ind_db,'IR_DP']=dados.iloc[row+2,col]
    db.loc[ind_db,'RendDP']=dados.iloc[row+4,col]

    db.loc[ind_db,'T_despar']=float(dados.iloc[row2,col2])
    db.loc[ind_db,'RSO_despar']=float(dados.iloc[row2+1,col2].split(':')[0].replace(',','.'))
    db.loc[ind_db,'Lav_despar']=float(dados.iloc[row2+2,col2].split(':')[0].replace(',','.'))
    
    bool1= db.loc[ind_db,'IV_DP']>50 and db.loc[ind_db,'IV_DP']<100
    bool2= db.loc[ind_db,'IR_DP']>1.4 and db.loc[ind_db,'IV_DP']<1.5
    bool3= db.loc[ind_db,'d_DP']>0.8 and db.loc[ind_db,'IV_DP']<90
    bool4= db.loc[ind_db,'RendDP']>10 and db.loc[ind_db,'RendR']<90
    bool5= db.loc[ind_db,'T_despar']>50 and db.loc[ind_db,'T_despar']<140
    bool6= db.loc[ind_db,'RSO_despar']>0 and db.loc[ind_db,'RSO_despar']<16
    bool7= db.loc[ind_db,'Lav_despar']>0 and db.loc[ind_db,'Lav_despar']<5

    if not bool1:
        print('erro no IV, desparafinado')
    if not bool2:
        print('erro no IR, desparafinado')
    if not bool3:
        print('erro no na densidade, desparafinado')
    if not bool4:
        print('erro no rendimento, desparafinado')
    if not bool5:
        print('erro na temp, desparafinado')
    if not bool6:
        print('erro no RSO, desparafinado')
    if not bool7:
        print('erro no lavagem, desparafinado')

    return db

#'Unnamed: 0', 'ORIGEM', 'Unnamed: 1', 'Visc_40C', 'Visc60_C',
#       'Visc80_C', 'Visc100_C', 'IV_C', 'd_C', 'IR_C', 'rend_raf', 'T_desaro',
#       'RSO_desaro', 'T_despar', 'RSO_despar', 'Lav_despar', 'Visc_40R',
#       'Visc_60R', 'Visc_80R', 'Visc100_R', 'IV_R', 'd_R', 'IR_R', 'RendR',
#       'RendDP', 'd_DP', 'IR_DP', 'Visc_40DP', 'PL_DP', 'Visc100_DP', 'IV_DP'
       
#dados_dp.iloc[101:110,:]
for i in range(25):       
    dados_dp=dados_dp.append(pd.Series(), ignore_index=True)

dados=SPB
dados_dp.loc[128,'ORIGEM']=dados.iloc[0,14]
dados_dp.loc[128,'Tipo oleo']='Bauna'
dados_dp.loc[128,'ORIGEM']=dados.iloc[0,14]
dados_dp.loc[128,'Tipo']='SPB'
dados_dp.loc[128,'Visc_40C']=np.nan
dados_dp.loc[128,'Visc60_C']=dados.iloc[7,7]
dados_dp.loc[128,'Visc100_C']=dados.iloc[8,8]
dados_dp.loc[128,'IV_C']=dados.iloc[9,14]
dados_dp.loc[128,'d_C']=dados.iloc[10,14]
dados_dp.loc[128,'IR_C']=dados.iloc[11,14]
dados_dp.loc[128,'T_desaro']=dados.iloc[3,15]
dados_dp.loc[128,'RSO_desaro']=float(dados.iloc[4,15].split(':')[0])
dados_dp.loc[128,'T_despar']=dados.iloc[15,14]
dados_dp.loc[128,'RSO_despar']=float(dados.iloc[16,14].split(':')[0])
dados_dp.loc[128,'Lav_despar']=float(dados.iloc[17,14].split(':')[0].replace(',','.'))
dados_dp.loc[128,'Visc_40R']=np.nan
dados_dp.loc[128,'Visc_60R']=dados.iloc[7,15]
dados_dp.loc[128,'Visc100_R']=dados.iloc[8,15]
dados_dp.loc[128,'IV_R']=dados.iloc[9,15]
dados_dp.loc[128,'d_R']=dados.iloc[10,15]
dados_dp.loc[128,'IR_R']=dados.iloc[11,15]
dados_dp.loc[128,'RendR']=dados.iloc[12,15]
dados_dp.loc[128,'Visc_40DP']=dados.iloc[20,14]
dados_dp.loc[128,'Visc100_DP']=dados.iloc[21,14]
dados_dp.loc[128,'IV_DP']=dados.iloc[22,14]
dados_dp.loc[128,'d_DP']=dados.iloc[23,14]
dados_dp.loc[128,'IR_DP']=dados.iloc[24,14]
dados_dp.loc[128,'RendDP']=dados.iloc[26,8]
#%% SPM
NL=sicol.parse('SPM')
dados=NL

'''
Ordem é ind_carga,ind_dados_da,ind_op_da,ind_dados_dp,ind_op_dp
'''
locations=[[[9,1],[9,2],[2,2],[22,2],[15,1]],
           [[9,3],[9,4],[2,4],[22,4],[15,4]],
           [[9,3],[9,5],[2,5],[22,5],[15,5]],
           [[9,3],[9,6],[2,6],[22,6],[15,6]],
           [[9,7],[9,8],[2,8],[22,8],[15,7]],
           [[9,7],[9,8],[2,9],[22,9],[15,9]],
           [[9,10],[9,11],[2,11],[22,10],[15,10]],
           [[9,12],[9,13],[2,13],[22,12],[15,12]],
           [[41,1],[41,2],[35,2],[52,1],[46,1]],
           [[41,3],[41,4],[35,4],[52,4],[46,4]],
           [[41,3],[41,5],[35,5],[52,5],[46,5]],
           [[41,3],[41,6],[35,6],[52,6],[46,6]],
           [[41,7],[41,8],[35,8],[52,7],[46,7]],#nesse aqui tem uma discordancia
           [[41,7],[41,9],[35,9],[52,9],[46,9]],#nesse aqui tem uma discordancia
           [[41,10],[41,11],[35,11],[52,10],[46,10]],
           [[41,12],[41,13],[35,13],[52,12],[46,12]],
           ]
ident=[[dados.columns[1],'SPM'],
       [dados.columns[3],'SPM'],
       [dados.columns[3],'SPM'],
       [dados.columns[3],'SPM'],
       [dados.columns[7],'SPM'],
       [dados.columns[7],'SPM'],
       [dados.columns[10],'SPM'],
       [dados.columns[12],'SPM'],
       [dados.iloc[33,1],'SPM'],
       [dados.iloc[33,3],'SPM'],
       [dados.iloc[33,3],'SPM'],
       [dados.iloc[33,3],'SPM'],
       [dados.iloc[33,7],'SPM'],
       [dados.iloc[33,7],'SPM'],
       [dados.iloc[33,10],'SPM'],
       [dados.iloc[33,12],'SPM'],
       ]
k=129
for i in range(len(locations)): 
    dados_dp=reorg_dados_carga(dados_dp,dados,k,locations[i][0],ident[i])
    dados_dp=reorg_dados_da(dados_dp,dados,k,locations[i][1],locations[i][2])
    dados_dp=reorg_dados_dp(dados_dp,dados,k,locations[i][3],locations[i][4])
    k=k+1
dados_dp['RendR']=dados_dp['RendR'].astype(float)
#for i in locations_col:
#    for j in locations_row:
#        if j==0:
#            dados_dp.loc[k,'ORIGEM']=dados.columns[i]
#            dados_dp.loc[k,'Unnamed: 1']='SPM'
#            dados_dp.loc[k,'Visc_40C']=dados.iloc[j+6,i]
#            dados_dp.loc[k,'Visc60_C']=dados.iloc[j+7,i]
#            dados_dp.loc[k,'Visc100_C']=dados.iloc[j+8,i]
#            dados_dp.loc[k,'IV_C']=dados.iloc[j+9,i]
#            dados_dp.loc[k,'d_C']=dados.iloc[j+10,i]
#            dados_dp.loc[k,'IR_C']=dados.iloc[j+11,i]
#            dados_dp.loc[k,'T_desaro']=dados.iloc[j+2,i+1]
#            dados_dp.loc[k,'RSO_desaro']=float(dados.iloc[j+3,i+1].split(':')[0])
#            dados_dp.loc[k,'T_despar']=dados.iloc[j+15,i]
#            dados_dp.loc[k,'RSO_despar']=float(dados.iloc[j+16,i].split(':')[0])
#            dados_dp.loc[k,'Lav_despar']=float(dados.iloc[j+17,i].split(':')[0].replace(',','.'))
#            dados_dp.loc[k,'Visc_60R']=dados.iloc[j+7,i+1]
#            dados_dp.loc[k,'Visc100_R']=dados.iloc[j+8,i+1]
#            dados_dp.loc[k,'IV_R']=dados.iloc[j+9,i+1]
#            dados_dp.loc[k,'d_R']=dados.iloc[j+10,i+1]
#            dados_dp.loc[k,'IR_R']=dados.iloc[j+11,i+1]
#            dados_dp.loc[k,'RendR']=dados.iloc[j+12,i+1]
#            dados_dp.loc[k,'Visc_40DP']=dados.iloc[j+20,i]
#            dados_dp.loc[k,'Visc100_DP']=dados.iloc[j+21,i]
#            dados_dp.loc[k,'IV_DP']=dados.iloc[j+22,i]
#            dados_dp.loc[k,'d_DP']=dados.iloc[j+23,i]
#            dados_dp.loc[k,'IR_DP']=dados.iloc[j+24,i]
#            dados_dp.loc[k,'RendDP']=dados.iloc[j+37,i]
#            k=k+1
#        else:
#            dados_dp.loc[k,'ORIGEM']=dados.columns[i]
#            dados_dp.loc[k,'Unnamed: 1']='SPM'
#            dados_dp.loc[k,'Visc_40C']=dados.iloc[j+6,i]
#            dados_dp.loc[k,'Visc60_C']=dados.iloc[j+7,i]
#            dados_dp.loc[k,'Visc100_C']=dados.iloc[j+8,i]
#            dados_dp.loc[k,'IV_C']=dados.iloc[j+9,i]
#            dados_dp.loc[k,'d_C']=dados.iloc[j+10,i]
#            dados_dp.loc[k,'IR_C']=dados.iloc[j+11,i]
#            dados_dp.loc[k,'T_desaro']=dados.iloc[j+2,i+1]
#            dados_dp.loc[k,'RSO_desaro']=float(dados.iloc[j+3,i+1].split(':')[0])
#            dados_dp.loc[k,'T_despar']=dados.iloc[j+13,i]
#            dados_dp.loc[k,'RSO_despar']=float(dados.iloc[j+14,i].split(':')[0])
#            dados_dp.loc[k,'Lav_despar']=float(dados.iloc[j+15,i].split(':')[0].replace(',','.'))
#            dados_dp.loc[k,'Visc_60R']=dados.iloc[j+7,i+1]
#            dados_dp.loc[k,'Visc100_R']=dados.iloc[j+8,i+1]
#            dados_dp.loc[k,'IV_R']=dados.iloc[j+9,i+1]
#            dados_dp.loc[k,'d_R']=dados.iloc[j+10,i+1]
#            dados_dp.loc[k,'IR_R']=dados.iloc[j+11,i+1]
#            dados_dp.loc[k,'RendR']=dados.iloc[j+12,i+1]
#            dados_dp.loc[k,'Visc_40DP']=dados.iloc[j+20,i]
#            dados_dp.loc[k,'Visc100_DP']=dados.iloc[j+21,i]
#            dados_dp.loc[k,'IV_DP']=dados.iloc[j+22,i]
#            dados_dp.loc[k,'d_DP']=dados.iloc[j+23,i]
#            dados_dp.loc[k,'IR_DP']=dados.iloc[j+24,i]
#            dados_dp.loc[k,'RendDP']=dados.iloc[j+27,i]
#            k=k+1

#%% NM
dados=sicol.parse('NM')
'''
Ordem é ind_carga,ind_dados_da,ind_op_da,ind_dados_dp,ind_op_dp
'''
locations=[[[11,22],[11,23],[5,23],[24,22],[17,22]],
           [[11,24],[11,25],[5,25],[24,24],[17,24]],
           [[11,26],[11,27],[5,27],[24,26],[17,26]],
           [[11,28],[11,29],[5,29],[24,28],[17,28]],
           ]

ident=[[dados.iloc[2,22],'NM'],
       [dados.iloc[2,24],'NM'],
       [dados.iloc[2,26],'NM'],
       [dados.iloc[2,28],'MM'],
       ]
k=145
for i in range(len(locations)): 
    dados_dp=reorg_dados_carga(dados_dp,dados,k,locations[i][0],ident[i])
    dados_dp=reorg_dados_da(dados_dp,dados,k,locations[i][1],locations[i][2])
    dados_dp=reorg_dados_dp(dados_dp,dados,k,locations[i][3],locations[i][4])
    k=k+1
dados_dp['RendR']=dados_dp['RendR'].astype(float)
#%% NL
dados=sicol.parse('NL')
'''
Ordem é ind_carga,ind_dados_da,ind_op_da,ind_dados_dp,ind_op_dp
'''
locations=[[[10,31],[10,32],[4,32],[23,31],[16,31]],
           [[10,33],[10,34],[4,34],[23,33],[16,33]],
           ]

ident=[[dados.iloc[1,31],'NL'],
       [dados.iloc[1,33],'NL'],
       ]
k=149
for i in range(len(locations)): 
    dados_dp=reorg_dados_carga(dados_dp,dados,k,locations[i][0],ident[i])
    dados_dp=reorg_dados_da(dados_dp,dados,k,locations[i][1],locations[i][2])
    dados_dp=reorg_dados_dp(dados_dp,dados,k,locations[i][3],locations[i][4])
    k=k+1
dados_dp['RendR']=dados_dp['RendR'].astype(float)

#%% NP
dados=sicol.parse('NP')
'''
Ordem é ind_carga,ind_dados_da,ind_op_da,ind_dados_dp,ind_op_dp
'''
locations=[[[9,5],[9,6],[3,6],[22,5],[15,5]],
           [[9,10],[9,11],[3,11],[22,10],[15,5]],
           [[9,12],[9,13],[3,13],[22,12],[15,12]],
           [[9,14],[9,15],[3,15],[22,14],[15,14]],
           ]

ident=[[dados.iloc[0,5],'NP'],
       [dados.iloc[0,10],'NP'],
       [dados.iloc[0,12],'NP'],
       [dados.iloc[0,14],'NP'],
       ]
k=151
for i in range(len(locations)): 
    dados_dp=reorg_dados_carga(dados_dp,dados,k,locations[i][0],ident[i])
    dados_dp=reorg_dados_da(dados_dp,dados,k,locations[i][1],locations[i][2])
    dados_dp=reorg_dados_dp(dados_dp,dados,k,locations[i][3],locations[i][4])
    k=k+1
dados_dp['RendR']=dados_dp['RendR'].astype(float)

dados_dp.to_csv(path_data+'\\dados_desparafinacao_20210711.csv')
#%% BS
sicol=pd.ExcelFile(path_data+'\\SICOL_20210712.xls')
dados=sicol.parse('BS')
dados_dp=pd.read_csv(path_data+'\\dados_desparafinacao_20210711.csv')
dados_dp.drop('Unnamed: 0',inplace=True,axis=1)
for i in range(20):       
    dados_dp=dados_dp.append(pd.Series(), ignore_index=True)

'''
Ordem é ind_carga,ind_dados_da,ind_op_da,ind_dados_dp,ind_op_dp
'''
locations=[[[9,1],[9,2],[2,2],[22,2],[15,2]],
           [[9,1],[9,3],[2,3],[22,3],[15,3]],
           [[9,4],[9,5],[2,5],[22,5],[15,5]],
           [[9,4],[9,6],[2,6],[22,6],[15,6]],
           [[9,9],[9,10],[2,10],[22,9],[15,9]],
           [[9,11],[9,12],[2,12],[22,11],[15,11]],
           [[9,13],[9,14],[2,14],[22,13],[15,13]],
           [[9,15],[9,16],[2,16],[22,15],[15,15]],
           [[9,17],[9,18],[2,18],[22,17],[15,17]],
           [[9,19],[9,20],[2,20],[22,19],[15,19]],
           ]

ident=[[dados.columns[1],'BS'],
       [dados.columns[1],'BS'],
       [dados.columns[4],'BS'],
       [dados.columns[4],'BS'],
       [dados.columns[9],'BS'],
       [dados.columns[11],'BS'],
       [dados.columns[13],'BS'],
       [dados.columns[15],'BS'],
       [dados.columns[17],'BS'],
       [dados.columns[19],'BS'],
       ]
k=155
for i in range(len(locations)): 
    dados_dp=reorg_dados_carga(dados_dp,dados,k,locations[i][0],ident[i])
    dados_dp=reorg_dados_da(dados_dp,dados,k,locations[i][1],locations[i][2])
    dados_dp=reorg_dados_dp(dados_dp,dados,k,locations[i][3],locations[i][4])
    k=k+1

dados_dp.to_csv(path_data+'\\dados_desparafinacao_20210712.csv',index=False)
#%%

path_data=r'E:\coppe\SICOL_novo\dados'
sicol=pd.read_excel(path_data+'\\NP_AL_FURFURAL.xls')
dados_da2=pd.read_csv(r'E:\coppe\SICOL_novo\dados\dados_desaromatizacao_20210708.csv')
dados_da=pd.DataFrame([],columns=dados_da2.columns)
       
for i in range(37):       
    dados_da=dados_da.append(pd.Series(), ignore_index=True)

bool_carga=sicol['Unnamed: 3']=='RAFINADO'
dados_da.loc[:37,'ORIGEM']=sicol.loc[bool_carga,'Unnamed: 4'].values
dados_da.loc[:37,'Tipo']=sicol.loc[bool_carga,'Unnamed: 1'].values
dados_da.loc[:37,'d_C']=sicol.loc[6,'Unnamed: 14']
dados_da.loc[:37,'IR_C']=sicol.loc[6,'Unnamed: 15']
dados_da.loc[:37,'T_desaro']=sicol.loc[bool_carga,'Unnamed: 5'].values
dados_da.loc[:37,'RSO_desaro']=sicol.loc[bool_carga,'Unnamed: 6'].values
dados_da.loc[:37,'Visc100_R']=sicol.loc[bool_carga,'Unnamed: 18'].values
dados_da.loc[:37,'d_R']=sicol.loc[bool_carga,'Unnamed: 14'].values
dados_da.loc[:37,'IR_R']=sicol.loc[bool_carga,'Unnamed: 15'].values
dados_da.loc[:37,'IV_R']=sicol.loc[bool_carga,'Unnamed: 19'].values
dados_da.loc[:37,'RendR']=sicol.loc[bool_carga,'Unnamed: 11'].values



path_data=r'E:\coppe\SICOL_novo\dados'
sicol=pd.read_excel(path_data+'\\NM_AL_FURFURAL.xls')

#dados_dp.iloc[101:110,:]
for i in range(36):       
    dados_da=dados_da.append(pd.Series(), ignore_index=True)



bool_carga=sicol['Unnamed: 3']=='RAFINADO'
dados_da.loc[37:73,'ORIGEM']=sicol.loc[bool_carga,'Unnamed: 4'].values
dados_da.loc[37:73,'Tipo']=sicol.loc[bool_carga,'Unnamed: 1'].values
dados_da.loc[37:73,'d_C']=sicol.loc[6,'Unnamed: 14']
dados_da.loc[37:73,'IR_C']=sicol.loc[6,'Unnamed: 15']
dados_da.loc[37:73,'T_desaro']=sicol.loc[bool_carga,'Unnamed: 5'].values
dados_da.loc[37:73,'RSO_desaro']=sicol.loc[bool_carga,'Unnamed: 6'].values
dados_da.loc[37:73,'Visc100_R']=sicol.loc[bool_carga,'Unnamed: 18'].values
dados_da.loc[37:73,'d_R']=sicol.loc[bool_carga,'Unnamed: 14'].values
dados_da.loc[37:73,'IR_R']=sicol.loc[bool_carga,'Unnamed: 15'].values
dados_da.loc[37:73,'IV_R']=sicol.loc[bool_carga,'Unnamed: 19'].values
dados_da.loc[37:73,'RendR']=sicol.loc[bool_carga,'Unnamed: 11'].values

path_data=r'E:\coppe\SICOL_novo\dados'
sicol=pd.read_excel(path_data+'\\NL_AL_FURFURAL.xls')

#dados_dp.iloc[101:110,:]
for i in range(36):       
    dados_da=dados_da.append(pd.Series(), ignore_index=True)



bool_carga=sicol['Unnamed: 3']=='RAFINADO'
dados_da.loc[73:,'ORIGEM']=sicol.loc[bool_carga,'Unnamed: 4'].values
dados_da.loc[73:,'Tipo']=sicol.loc[bool_carga,'Unnamed: 1'].values
dados_da.loc[73:,'d_C']=sicol.loc[6,'Unnamed: 14']
dados_da.loc[73:,'IR_C']=sicol.loc[6,'Unnamed: 15']
dados_da.loc[73:,'T_desaro']=sicol.loc[bool_carga,'Unnamed: 5'].values
dados_da.loc[73:,'RSO_desaro']=sicol.loc[bool_carga,'Unnamed: 6'].values
dados_da.loc[73:,'Visc100_R']=sicol.loc[bool_carga,'Unnamed: 18'].values
dados_da.loc[73:,'d_R']=sicol.loc[bool_carga,'Unnamed: 14'].values
dados_da.loc[73:,'IR_R']=sicol.loc[bool_carga,'Unnamed: 15'].values
dados_da.loc[73:,'IV_R']=sicol.loc[bool_carga,'Unnamed: 19'].values
dados_da.loc[73:,'RendR']=sicol.loc[bool_carga,'Unnamed: 11'].values

tipos=[['DEST. NP','DEST. NL','DEST. NM'],['NP', 'NM', 'NL']]
for i in range(3):
    temp1=tipos[0][i]
    temp2=tipos[1][i]
    sorted_da=dados_da.loc[dados_da['Tipo']==temp1,:].sort_values(by=['T_desaro','RSO_desaro'])
    sorted_da2=dados_da2.loc[dados_da2['Tipo']==temp2,:].sort_values(by=['T_desaro','RSO_desaro'])
    num1=sorted_da.loc[:,['IV_R', 'd_R', 'IR_R', 'RendR']]
    num2=sorted_da2.loc[:,['IV_R', 'd_R', 'IR_R', 'RendR']]
    print(np.sum((num2-num1)**2))
i=0
#dados_da.to_csv(path_data+r'\\dados_desaromatizacao_20210601.csv',index=False)

