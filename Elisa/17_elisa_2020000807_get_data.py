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
path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_07082020'

files= os.listdir(path_image)
images=[]
image = cv2.imread(path_image+'\\'+'placa1 HP.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'placa1'])
image = cv2.imread(path_image+'\\'+'placa2 HP.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images.append([image.copy(),'placa2'])

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

coords=[[(144.5,107.7),(147.165,604.997),(910,603.2)],[(129,114.5),(130,612.5),(898,613)]]

# convert the (x, y) coordinates and radius of the circles to integers
# loop over the (x, y) coordinates and radius of the circles
for (x, y, r) in circles:
	# draw the circle in the output image, then draw a rectangle
	# corresponding to the center of the circle
	cv2.circle(output, (x, y), r, (0, 255, 0), 4)
	cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
plt.figure();plt.imshow(images[0][0])


coords=np.array([[(144.5,107.7),(147.165,604.997),(910,603.2)],[(129,114.5),(130,612.5),(898,613)]])
circles_collect=[]    
for j in range(2):
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
    plt.figure();plt.imshow(output)
    circles_collect.append(circles_num)
image_HSV=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
plt.figure();plt.imshow(image_HSV[:,:,0])
#guardar isso pra depois

intensity_collect=[]
for i in range(2):
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


import pandas as pd

planilha_1 = pd.read_excel(path_image+'//'+'Espetrografia_Elisa.xlsx',sheet_name='Controle positivo')
planilha_2 = pd.read_excel(path_image+'//'+'Espetrografia_Elisa.xlsx',sheet_name='Brancos')
planilha=planilha_1.iloc[22:30,2:].values-planilha_1.iloc[32:,2:].values
planilha_2b=planilha_2.iloc[22:30,2:].values-planilha_2.iloc[32:,2:].values
planilha=pd.DataFrame(planilha)
planilha2=pd.DataFrame(planilha_2b)
    #planilha=pd.read_excel(path_image+'//'+dict_photo_planilha[images[i][1]],sheet_name='Result data')

#    planilha=pd.read_excel(path_image+'//'+dict_photo_planilha[images[i][1]],sheet_name='Result data')
DF_collect=[]
for i in range(2):
    DF_dados=pd.DataFrame(intensity_collect[i],columns=['X pos','Y pos','sat median','sat mean', 'sat IQR'])
    DF_dados['absorb']=0
    DF_collect.append(DF_dados.copy())
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
