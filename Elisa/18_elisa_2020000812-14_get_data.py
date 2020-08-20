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
import pandas as pd

path_image1= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_120820'
path_image2= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_130820'
path_image3= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_140820'

images=[]
image = cv2.imread(path_image1+'\\'+'HP_08122020.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'placa_0812'])
image = cv2.imread(path_image2+'\\'+'HP_Placa1 130820 001.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'placa1_1308'])
image = cv2.imread(path_image2+'\\'+'HP_Placa2 130820 001.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'placa2_1308'])
image = cv2.imread(path_image3+'\\'+'Placa1_14082020.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'placa1_1408'])
image = cv2.imread(path_image3+'\\'+'Placa2_14082020.tif',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'placa2_1408'])

plt.imshow(image)

circles_num=np.zeros((96,3))
gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50)
circles=circles[:,circles[0,:,2]<100,:]
output = image.copy()

circles = np.round(circles[0, :]).astype("int")
circles=circles[circles[:,2]<80,:]
output = image.copy()
image2 = image.copy()


# convert the (x, y) coordinates and radius of the circles to integers
# loop over the (x, y) coordinates and radius of the circles
for (x, y, r) in circles:
	# draw the circle in the output image, then draw a rectangle
	# corresponding to the center of the circle
	cv2.circle(output, (x, y), r, (0, 255, 0), 4)
	cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
plt.figure();plt.imshow(output)

for i in images:
    plt.figure();plt.imshow(i[0]);plt.title(i[1])
    
coords=np.array([[(127.71,99.735),(131.5,596.9),(894.37,595.97)],
                  [(128.4,92.88),(129.97,591.014),(891.729,590.22)],
                  [(136.07,101.66),(136.87,599.471),(897.48,595.5)],
                  [(133.98,101.579),(131.1,601.25),(895.94,599.44)],
                  [(128.45,84.91),(130.36,581.53),(896.23,581.83)]])
#                  [(132.68,84.91),(129.89,587.28),(896.03,583.55)],

circles_collect=[]    
for j in range(5):
    circles_num=np.zeros((96,3))
    d_vert=(coords[j][1]-coords[j][0])/7
    d_horz=(coords[j][2]-coords[j][1])/11
    init_pos=coords[j][0]
    pos=init_pos.copy()
    r=np.linalg.norm(d_horz)*0.35;circles_num[:,2]=r
        
    for i in range(96):
        circles_num[i,[0,1]]=pos
        pos=pos+d_horz
        if np.mod(i,12)==11:
            pos=init_pos.copy()
            pos=pos+d_vert*np.ceil(i/12)        
    circles_num=circles_num.astype(int)
    output=images[j][0].copy()
    for (x, y, r) in circles_num:
        x=x.astype(int);    y=y.astype(int);    r=r.astype(int)
    
        cv2.circle(output, (x, y), r, (0, 255, 0), 10)
        cv2.rectangle(output, (x - 7, y - 7), (x + 7, y + 7), (0, 128, 255), -1)    
    plt.figure();plt.imshow(output);plt.title(images[j][1])
    circles_collect.append(circles_num)
image_HSV=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
plt.figure();plt.imshow(image_HSV[:,:,0])
#guardar isso pra depois

intensity_collect=[]
for i in range(5):
    intensity=np.zeros((96,5))
    circles_num=circles_collect[i].copy()
    image_HSV=cv2.cvtColor(images[i][0], cv2.COLOR_RGB2HSV)
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
    intensity_collect.append(intensity)



planilha_1 = pd.read_excel(path_image1+'//'+'08.12.2020 - Leitura ELISA.xlsx')
planilha_2 = pd.read_excel(path_image2+'//'+'08.13.20 Leitura ELISA.xlsx',sheet_name='Placa 1')
planilha_3 = pd.read_excel(path_image2+'//'+'08.13.20 Leitura ELISA.xlsx',sheet_name='Placa 2')
planilha_4 = pd.read_excel(path_image3+'//'+'08.14.20 Leitura ELISA.xlsx',sheet_name='Placa 1')
planilha_5 = pd.read_excel(path_image3+'//'+'08.14.20 Leitura ELISA.xlsx',sheet_name='Placa 2')
planilha_1=planilha_1.iloc[32:,2:14]
planilha_2=planilha_2.iloc[32:,2:14]
planilha_3=planilha_3.iloc[32:,2:14]
planilha_4=planilha_4.iloc[32:,2:14]
planilha_5=planilha_5.iloc[32:,2:14]
planilhas=[planilha_1,planilha_2,planilha_3,planilha_4,planilha_5]

DF_collect=[]
for i in range(5):
    DF_dados=pd.DataFrame(intensity_collect[i],columns=['X pos','Y pos','sat median','sat mean', 'sat IQR'])
    DF_dados['absorb']=0
    ind1=np.argsort(intensity[:,0])
    int_sorted1=intensity[ind1,:].copy()
    int_norm1=np.round((intensity[:,0]-np.min(intensity[:,0]))/((np.max(intensity[:,0])-np.min(intensity[:,0]))/11))
    int_norm2=np.round((intensity[:,1]-np.min(intensity[:,1]))/((np.max(intensity[:,1])-np.min(intensity[:,1]))/7))
    for j in range(intensity.shape[0]):
        DF_dados.iloc[j,5]=planilhas[i].iloc[int_norm2[j].astype(int),11-int_norm1[j].astype(int)]
    DF_dados['log absorb']=np.log(DF_dados['absorb'])
    DF_dados['sqrt absorb']=np.sqrt(DF_dados['absorb'])
    print(DF_dados.corr()['absorb'])
    DF_collect.append(DF_dados.copy())
    DF_dadsp.plot(x='sat median',y='absorb',style='+')
    DF_dados.to_csv(path_image1+'\\'+images[i][1]+'.csv')    
    
for i in range(5):
    DF_collect[i].plot(x='Y pos',y='absorb',style='+')
