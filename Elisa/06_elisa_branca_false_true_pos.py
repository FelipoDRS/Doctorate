# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:13:55 2020

@author: Felipo Soares
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
path_planilha=r'E:\Downloads_HD\AnaliseImagem-ELISA\EspectroFotometro'

DF_dados=pd.read_csv(path_planilha+'\\placa_branca_230626.csv',index_col=0)

bool_dil1=np.logical_or(np.logical_and(DF_dados['X pos']>=230,DF_dados['X pos']<=255),np.logical_and(DF_dados['X pos']>=786,DF_dados['X pos']<=811))
dil1=DF_dados.loc[bool_dil1,:]

bool_dil2=np.logical_or(np.logical_and(DF_dados['X pos']>=369,DF_dados['X pos']<=394),np.logical_and(DF_dados['X pos']>=925,DF_dados['X pos']<=950))
dil2=DF_dados.loc[bool_dil2,:]

bool_dil3=np.logical_or(np.logical_and(DF_dados['X pos']>=508,DF_dados['X pos']<=533),np.logical_and(DF_dados['X pos']>=1065,DF_dados['X pos']<=1089))
dil3=DF_dados.loc[bool_dil3,:]

bool_dil4=np.logical_or(np.logical_and(DF_dados['X pos']>=647,DF_dados['X pos']<=672),np.logical_and(DF_dados['X pos']>=1204,DF_dados['X pos']<=1228))
dil4=DF_dados.loc[bool_dil4,:]

neg=np.zeros((24,))
neg[2::3]=1
pos=1-neg

neg_all=np.zeros((96,))
neg_all[2::3]=1
pos_all=1-neg


pos_dil1=dil1.loc[pos.astype(bool)];neg_dil1=dil1.loc[neg.astype(bool)]
pos_dil2=dil2.loc[pos.astype(bool)];neg_dil2=dil2.loc[neg.astype(bool)]
pos_dil3=dil3.loc[pos.astype(bool)];neg_dil3=dil3.loc[neg.astype(bool)]
pos_dil4=dil4.loc[pos.astype(bool)];neg_dil4=dil4.loc[neg.astype(bool)]

plt.figure();plt.plot(pos_dil1['sat'],pos_dil1['log absorb'],'b+')
plt.plot(neg_dil1['sat'],neg_dil1['log absorb'],'r+')

plt.plot(pos_dil2['sat'],pos_dil2['log absorb'],'b+')
plt.plot(neg_dil2['sat'],neg_dil2['log absorb'],'r+')
plt.plot(pos_dil3['sat'],pos_dil3['log absorb'],'b+')
plt.plot(neg_dil3['sat'],neg_dil3['log absorb'],'r+')
plt.plot(pos_dil4['sat'],pos_dil4['log absorb'],'b+')
plt.plot(neg_dil4['sat'],neg_dil4['log absorb'],'r+')