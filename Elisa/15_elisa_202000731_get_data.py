# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:13:55 2020

@author: Felipo Soares
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os
path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_290720'

files= os.listdir(path_image)
images=[]
image = cv2.imread(path_image+'\\'+'placa290720 002 - HP 001.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'hp'])

image_rot=ndimage.rotate(image,-90)
plt.imshow(image_rot)

circles_num=np.zeros((96,3))

coords=np.array([[ 102.965,  83.1086],
       [887.738,  84.9951],
       [ 107.681, 571.706]])
    

circles_num=np.zeros((96,3))
d_vert=(coords[2]-coords[0])/7
d_horz=(coords[1]-coords[0])/11
init_pos=coords[0]
pos=init_pos.copy()
r=np.linalg.norm(d_horz)*0.35;circles_num[:,2]=r
    
for i in range(96):
    circles_num[i,[0,1]]=pos
    pos=pos+d_horz
    if np.mod(i,12)==11:
        pos=init_pos.copy()
        pos=pos+d_vert*np.ceil(i/12)        
circles_num=circles_num.astype(int)

output=image_rot.copy()
for (x, y, r) in circles_num:
    x=x.astype(int);    y=y.astype(int);    r=r.astype(int)

    cv2.circle(output, (x, y), r, (0, 255, 0), 10)
    cv2.rectangle(output, (x - 7, y - 7), (x + 7, y + 7), (0, 128, 255), -1)    
plt.figure();plt.imshow(output)

image_HSV=cv2.cvtColor(image_rot, cv2.COLOR_RGB2HSV)

#guardar isso pra depois

intensity=np.zeros((96,5))
for j in range(circles_num.shape[0]):
    r=circles_num[0,2].astype(int)
    collect_int=np.zeros(4*r*r)
    collect_int[:]=np.nan
    count=0
    for k in range(circles_num[j,1]-r,circles_num[j,1]+r):
        for l in range(circles_num[j,0]-r,circles_num[j,0]+r):
            if (k-circles_num[j,1])**2 +(l-circles_num[j,0])**2 <= r*r:
                 collect_int[count]=image_HSV[k,l,1]
                 count=count+1
    
    intensity[j,0]=circles_num[j,0];intensity[j,1]=circles_num[j,1];
    intensity[j,2]=np.nanmedian(collect_int);intensity[j,3]=np.nanmean(collect_int);
    intensity[j,4]=(np.nanquantile(collect_int,0.25)+np.nanquantile(collect_int,0.75))/2
    
print(np.mean(np.isnan(collect_int)))



import pandas as pd

DF_collect=[]
planilha = pd.read_excel(path_image+'//'+'Leituras ELISA.xlsx')
planilha=planilha.iloc[32:,2:-1]

    #planilha=pd.read_excel(path_image+'//'+dict_photo_planilha[images[i][1]],sheet_name='Result data')

#    planilha=pd.read_excel(path_image+'//'+dict_photo_planilha[images[i][1]],sheet_name='Result data')
DF_dados=pd.DataFrame(intensity,columns=['X pos','Y pos','sat median','sat mean', 'sat IQR'])
DF_dados['absorb']=0

ind1=np.argsort(intensity[:,0])
int_sorted1=intensity[ind1,:].copy()
int_norm1=np.round((intensity[:,0]-np.min(intensity[:,0]))/((np.max(intensity[:,0])-np.min(intensity[:,0]))/11))
int_norm2=np.round((intensity[:,1]-np.min(intensity[:,1]))/((np.max(intensity[:,1])-np.min(intensity[:,1]))/7))
for j in range(intensity.shape[0]):
    DF_dados.iloc[j,5]=planilha.iloc[int_norm2[j].astype(int),11-int_norm1[j].astype(int)]
DF_dados['log absorb']=np.log(DF_dados['absorb'])
DF_dados['sqrt absorb']=np.sqrt(DF_dados['absorb'])

DF_dados.to_csv(path_image+'\\dados_20200731.csv')    
    
plt.figure();plt.imshow(images_HSV[1][:,:,1]);plt.colorbar()

DF_dados.plot(x=2,y='absorb',style='+')

DF_dados.plot(x=0,y=2,style='+')
DF_dados.plot(x=1,y=2,style='+')
DF_dados.plot(x=2,y='log absorb',style='+')
