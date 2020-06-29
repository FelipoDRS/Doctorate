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
    if '.jpg' in i and 'Branca' in i and 'Amarelo' in i:    
        image = cv2.imread(path_image+'\\'+i,1)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append([image.copy(),i])
grays=[]
circles=[]
for i in images:
    gray=cv2.cvtColor(i[0], cv2.COLOR_RGB2GRAY)
    grays.append(gray.copy())
    circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50)
    circle=circle[:,circle[0,:,2]<100,:]
    circle = np.round(circle[0, :]).astype("int")
    circles.append(circle.copy())

output = image.copy()

# convert the (x, y) coordinates and radius of the circles to integers
# loop over the (x, y) coordinates and radius of the circles
for (x, y, r) in circle:
	# draw the circle in the output image, then draw a rectangle
	# corresponding to the center of the circle
	cv2.circle(output, (x, y), r, (0, 255, 0), 4)
	cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)    
plt.figure();plt.imshow(output)

#rotate image, later add tests

us=[]
angs=[]
v=np.array([0,1])
for i in circles:
    ind1=np.argsort(i[:,0])
    circles_sorted1=i[ind1,:].copy()
    ind2=np.argsort(i[:,1])
    circles_sorted2=i[ind2,:].copy()
    u=circles_sorted1[5,[0,1]]-circles_sorted1[0,[0,1]]
    ang=180-np.arccos(np.vdot(v,u)/np.linalg.norm(u))*360/(2*np.pi)
    us.append(u.copy())
    angs.append(ang.copy())

from scipy import ndimage
grays_rot=[]
images_rot=[]
for i in range(len(angs)):
    gray2 = ndimage.rotate(grays[i], angs[i], reshape=False)
    output2 = ndimage.rotate(images[i][0], angs[i], reshape=False)
    grays_rot.append(gray2);images_rot.append(output2)
    plt.figure();plt.imshow(gray2,cmap='gray')

for i in grays_rot:
    circle2 = cv2.HoughCircles(i, cv2.HOUGH_GRADIENT, 1.2, 40)
    circle2=circle2[:,circle2[0,:,2]<100,:]
    circle2 = np.round(circle2[0, :]).astype("int")
    print(circle2.shape)
    output2=i.copy()
    for (x, y, r) in circle2:
        cv2.circle(output2, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    plt.figure();plt.imshow(output2)

plt.figure()
plt.imshow(gray2,cmap='gray')
output3 = ndimage.rotate(image, ang, reshape=False)
image_HSV=cv2.cvtColor(output3, cv2.COLOR_RGB2HSV)

lower = np.array([22, 20, 0], dtype="uint8")
upper = np.array([65, 255, 255], dtype="uint8")
mask = cv2.inRange(image_HSV, lower, upper)

plt.figure();plt.imshow(mask)

plt.figure();plt.imshow(image_HSV[:,:,0]);plt.colorbar()
plt.figure();plt.imshow(image_HSV[:,:,1]);plt.colorbar()
plt.figure();plt.imshow(image_HSV[:,:,2]);plt.colorbar()

#guardar isso pra depois
image2=image.copy()
image2[:,:,[0,1]]=0
plt.figure();plt.imshow(image2)

intensity=np.zeros((96,3))
for i in range(circles.shape[0]):
    collect_int=np.zeros(4*r*r)
    collect_int[:]=np.nan
    l=0
    for j in range(circles[i,1]-r,circles[i,1]+r):
        for k in range(circles[i,0]-r,circles[i,0]+r):
            if (j-circles[i,1])**2 +(k-circles[i,0])**2 <= r*r:
                 collect_int[l]=image_HSV[j,k,1]
                 l=l+1
    print(np.mean(np.isnan(collect_int)))
    intensity[i,0]=circles[i,0];intensity[i,1]=circles[i,1];
    intensity[i,2]=np.nanmean(collect_int)

import pandas as pd
path_planilha=r'E:\Downloads_HD\AnaliseImagem-ELISA\EspectroFotometro'
planilha=pd.read_excel(path_planilha+'\\ELISA 230620.xlsx',sheet_name='Placa branca',header=5)
ind1=np.argsort(intensity[:,0])
int_sorted1=intensity[ind1,:].copy()
int_norm1=np.round((intensity[:,0]-np.min(intensity[:,0]))/((np.max(intensity[:,0])-np.min(intensity[:,0]))/7))
int_norm2=np.round((intensity[:,1]-np.min(intensity[:,1]))/((np.max(intensity[:,1])-np.min(intensity[:,1]))/11))

DF_dados=pd.DataFrame(intensity)
DF_dados['absorb']=0

for i in range(intensity.shape[0]):
    DF_dados.iloc[i,3]=planilha.iloc[7-int_norm1[i].astype(int),11-int_norm2[i].astype(int)]
    
DF_dados.plot(x=2,y='absorb',style='+')
DF_dados['log absorb']=np.log(DF_dados['absorb'])

print(DF_dados.corr())
DF_dados.plot(x=0,y=2,style='+')
DF_dados.plot(x=1,y=2,style='+')
DF_dados.plot(x=2,y='log absorb',style='+')
