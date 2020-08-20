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
path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_250620'

files= os.listdir(path_image)
images=[]
for i in files:
    if '.jpg' in i:    
        image = cv2.imread(path_image+'\\'+i,1)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append([image.copy(),i])
grays=[]
circles=[]
for i in images:
    gray=cv2.cvtColor(i[0], cv2.COLOR_RGB2GRAY)
    grays.append(gray.copy())
    circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 80,param1=100,param2=30, minRadius=35,maxRadius=60)
    circle=circle[:,circle[0,:,2]<100,:]
    circle = np.round(circle[0, :]).astype("int")
    circles.append(circle.copy())
    output = i[0].copy()
    for (x, y, r) in circle:
    
	# draw the circle in the output image, then draw a rectangle
	# corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)    
    plt.figure();plt.imshow(output)
    #plt.pause(15)

plt.close()
#rotate image, later add tests
images_HSV=[]
images_HLS=[]

for i in images:
    images_HSV.append(cv2.cvtColor(i[0], cv2.COLOR_RGB2HSV))
    images_HLS.append(cv2.cvtColor(i[0], cv2.COLOR_RGB2HLS))
import time
#for i in range(len(images_HSV)):
##    plt.figure();plt.imshow(images_HSV[i][:,:,0]);plt.colorbar()
    plt.figure();plt.imshow(images_HLS[1][:,:,0]);plt.colorbar()
    plt.figure();plt.imshow(images_HLS[1][:,:,1]);plt.colorbar()
    plt.figure();plt.imshow(images_HLS[1][:,:,2]);plt.colorbar()
# #   plt.figure();plt.imshow(images_HSV[i][:,:,2]);plt.colorbar()
#    plt.pause(15)
#    print(i)
#guardar isso pra depois

r=50
intensity_collect=[]
for i in range(len(images_HSV)):
    intensity=np.zeros((96,4))
    for j in range(circles[i].shape[0]):
        r=circles[i][0,2]
        collect_int=np.zeros((4*r*r,2))
        collect_int[:]=np.nan
        count=0
        for k in range(circles[i][j,1]-r,circles[i][j,1]+r):
            for l in range(circles[i][j,0]-r,circles[i][j,0]+r):
                if (k-circles[i][j,1])**2 +(l-circles[i][j,0])**2 <= r*r:
                     collect_int[count,0]=images_HSV[i][k,l,1]
                     collect_int[count,1]=images_HLS[i][k,l,2]
                     count=count+1
        
        intensity[j,0]=circles[i][j,0];intensity[j,1]=circles[i][j,1];
        intensity[j,[2,3]]=np.nanmean(collect_int,axis=0)
        
    intensity_collect.append(intensity.copy())
    print(np.mean(np.isnan(collect_int)))



import pandas as pd
xls = pd.ExcelFile(path_image+'\\Placas_de_ELISA_Absorbancia_25062020.xlsx')
sheets = xls.sheet_names

dict_photo_planilha={'Placa1_Transp_Canto_Amarelo.jpg':'Placa1_Transparente Amarelo',
'Placa1_Transp_Canto_Azul.jpg':'Placa1_Transparente Azul',
'Placa1_Transp_centro_Amarelo.jpg':'Placa1_Transparente Amarelo',
'Placa1_Transp_centro_Azul.jpg':'Placa1_Transparente Azul',
'Placa2_Branca_Canto_Amarelo.jpg': 'Placa2_Branca Amarelo',
'Placa2_Branca_Canto_Azul.jpg':'Placa2_Branca Azul',
'Placa2_Branca_Centro_Amarelo.jpg': 'Placa2_Branca Amarelo',
'Placa2_Branca_Centro_Azul.jpg':'Placa2_Branca Azul',
'Placa3_Branca_Canto_Amarelo.jpg':'Placa3_Branca Amarelo',
'Placa3_Branca_Canto_Azul.jpg': 'Placa3_Branca azul',
'Placa3_Branca_Centro_Amarelo.jpg':'Placa3_Branca Amarelo',
'Placa3_Branca_Centro_Azul.jpg': 'Placa3_Branca azul',
'Placa4_Tranp_OPD.jpg': 'Placa4_Transparente OPD',
'Placa5_Transp_OPD.jpg': 'Placa5_Transparente OPD'}
DF_collect=[]
for i in range(len(images_HSV)):
    intensity=intensity_collect[i]
    
    planilha=xls.parse(sheet_name=dict_photo_planilha[images[i][1]])
    DF_dados=pd.DataFrame(intensity,columns=['X pos','Y pos','sat','cor'])
    DF_dados['absorb']=0

    ind1=np.argsort(intensity[:,0])
    int_sorted1=intensity[ind1,:].copy()
    int_norm1=np.round((intensity[:,0]-np.min(intensity[:,0]))/((np.max(intensity[:,0])-np.min(intensity[:,0]))/11))
    int_norm2=np.round((intensity[:,1]-np.min(intensity[:,1]))/((np.max(intensity[:,1])-np.min(intensity[:,1]))/7))
    for j in range(intensity.shape[0]):
        DF_dados.iloc[j,3]=planilha.iloc[int_norm2[j].astype(int),11-int_norm1[j].astype(int)]
    DF_dados['log absorb']=np.log(DF_dados['absorb'])
    DF_dados['sqrt absorb']=np.sqrt(DF_dados['absorb'])
    
    DF_collect.append(DF_dados.copy())

l=0; 
for k in DF_collect:
#    k.to_csv(path_image+'\\20200307' +images[l][1].replace('.jpg','.csv'))
#    k.plot(x=2,y='sqrt absorb',style='+')
    plt.title(images[l][1])
    print(images[l][1])
    l=l+1
    print(k.corr()['sat'])
    
    
plt.figure();plt.imshow(images_HSV[1][:,:,1]);plt.colorbar()

DF_dados.plot(x=2,y='absorb',style='+')

DF_dados.plot(x=0,y=2,style='+')
DF_dados.plot(x=1,y=2,style='+')
DF_dados.plot(x=2,y='log absorb',style='+')
