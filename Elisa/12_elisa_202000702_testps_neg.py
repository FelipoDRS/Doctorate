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
path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_20200702'

files= os.listdir(path_image)
images=[]
for i in files:
    if '.jpeg' in i:    
        image = cv2.imread(path_image+'\\'+i,1)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append([image.copy(),i])


#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(images[2][0][:,:,2],cmap='gray')
#coords=plt.ginput(n=3,timeout=0,show_clicks=True)
#coords=np.array(coords)
#print(coords)
circles_num=np.zeros((96,3))

coords0=np.array([[ 459.58064516,  433.80080645],
       [2794.87029781,  397.77272727],
       [ 483.93769592, 1913.08424765]])
    
coords1=np.array([[522.31208016,  401.75498209],
 [2893.3809897,   362.07182042],
 [ 532.23287058, 1916.32898567]])
coords2=np.array([[ 267.97100313 , 250.77351097],
 [1448.76757725,  245.5255262 ],
 [ 266.72978551 ,1010.33773179]])
    
    
coords3=np.array([[ 392.98768473,  376.39666368],
       [2763.47301836,  369.3937528 ],
       [ 396.48914017, 1899.52978056]])

coords=[coords0,coords1,coords2,coords3]
circles_collect=[]
for i in range(len(images)):
    circles_num=np.zeros((96,3))
    coord=coords[i]    
    d_vert=(coord[2]-coord[0])/7
    d_horz=(coord[1]-coord[0])/11
    init_pos=coord[0]
    pos=init_pos.copy()
    r=np.linalg.norm(d_horz)*0.35;circles_num[:,2]=r
    
    for i in range(96):
        circles_num[i,[0,1]]=pos
        pos=pos+d_horz
        if np.mod(i,12)==11:
            pos=init_pos.copy()
            pos=pos+d_vert*np.ceil(i/12)        
    circles_collect.append(circles_num.astype(int).copy())


output=image.copy()
for (x, y, r) in circles_num:
    x=x.astype(int);    y=y.astype(int);    r=r.astype(int)

    cv2.circle(output, (x, y), r, (0, 255, 0), 15)
    cv2.rectangle(output, (x - 17, y - 17), (x + 17, y + 17), (0, 128, 255), -1)    
plt.figure();plt.imshow(output)

plt.close()
#rotate image, later add tests
images_HSV=[]
for i in images:
    images_HSV.append(cv2.cvtColor(i[0], cv2.COLOR_RGB2HSV))

for i in range(len(images_HSV)):
    a=images_HSV[i][:,:,0].copy()
    plt.figure();plt.imshow(a.astype(int));plt.colorbar()
    plt.figure();plt.imshow(images_HSV[i][:,:,1]);plt.colorbar()
#    plt.figure();plt.imshow(images_HSV[i][:,:,2]);plt.colorbar()
    plt.pause(15)
    print(i)
#guardar isso pra depois

intensity_collect=[]
for i in range(len(images_HSV)):
    intensity=np.zeros((96,5))
    for j in range(circles_collect[i].shape[0]):
        r=circles_collect[i][0,2].astype(int)
        collect_int=np.zeros(4*r*r)
        collect_int[:]=np.nan
        count=0
        for k in range(circles_collect[i][j,1]-r,circles_collect[i][j,1]+r):
            for l in range(circles_collect[i][j,0]-r,circles_collect[i][j,0]+r):
                if (k-circles_collect[i][j,1])**2 +(l-circles_collect[i][j,0])**2 <= r*r:
                     collect_int[count]=images_HSV[i][k,l,1]
                     count=count+1
        
        intensity[j,0]=circles_collect[i][j,0];intensity[j,1]=circles_collect[i][j,1];
        intensity[j,2]=np.nanmedian(collect_int);intensity[j,3]=np.nanmean(collect_int);
        intensity[j,4]=(np.nanquantile(collect_int,0.25)+np.nanquantile(collect_int,0.75))/2
        
    intensity_collect.append(intensity.copy())
    print(np.mean(np.isnan(collect_int)))



import pandas as pd

dict_photo_planilha={'placa_1_hemorio.jpeg':'p1 hemorio.xlsx',
'placa_1_hemorio600.jpeg':'p1 hemorio.xlsx',
'placa_2_hemorio300.jpeg':'p2 hemorio.xlsx',
'placa_2_hemorio600.jpeg':'p2 hemorio.xlsx'}
DF_collect=[]
for i in range(len(images_HSV)):
    intensity=intensity_collect[i].copy()
    xls = pd.ExcelFile(path_image+'//'+dict_photo_planilha[images[i][1]])

    #planilha=pd.read_excel(path_image+'//'+dict_photo_planilha[images[i][1]],sheet_name='Result data')
    planilha=xls.parse(sheet_name='Result Data',header=1,index_col=0)

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
    DF_collect.append(DF_dados.copy())

l=0; 
for k in DF_collect:
    k.to_csv(path_image+'\\b_' +images[l][1].replace('.jpeg','.csv'))
    k.plot(x=2,y='sqrt absorb',style='+')
    plt.title(images[l][1])
    print(images[l][1])
    l=l+1
    print(k.corr()[['sat median','sat mean','sat IQR']])
    
    
plt.figure();plt.imshow(images_HSV[1][:,:,1]);plt.colorbar()

DF_dados.plot(x=2,y='absorb',style='+')

DF_dados.plot(x=0,y=2,style='+')
DF_dados.plot(x=1,y=2,style='+')
DF_dados.plot(x=2,y='log absorb',style='+')
