# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:13:55 2020

@author: Felipo Soares
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Scanner'
image = cv2.imread(path_image+'\\Amarelo_placa-transp_230620-001_scanner(1).jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.figure();plt.imshow(image)

gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image2=image.copy()
image2[:,:,[0,1]]=0
gray2=cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
circles = cv2.HoughCircles(gray2*7, cv2.HOUGH_GRADIENT, 1.2, 30)
circles = np.round(circles[0, :]).astype("int")
circles=circles[circles[:,2]<60,:]
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



ind1=np.argsort(circles[:,0])
circles_sorted1=circles[ind1,:].copy()

ind2=np.argsort(circles[:,1])
circles_sorted2=circles[ind2,:].copy()

u=circles_sorted1[7,[0,1]]-circles_sorted1[0,[0,1]]
v=np.array([0,1])


ang=180-np.arccos(np.vdot(v,u)/np.linalg.norm(u))*360/(2*np.pi)
from scipy import ndimage

gray2 = ndimage.rotate(gray, -ang, reshape=False)
output2 = ndimage.rotate(image, -ang, reshape=False)
plt.figure();plt.imshow(gray2,cmap='gray')

circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1.2, 50)
circles = np.round(circles[0, :]).astype("int")
circles=circles[circles[:,2]<60,:]
output = image.copy()

# convert the (x, y) coordinates and radius of the circles to integers
# loop over the (x, y) coordinates and radius of the circles
output2 = ndimage.rotate(image, -ang, reshape=False)
for (x, y, r) in circles:
	# draw the circle in the output image, then draw a rectangle
	# corresponding to the center of the circle
	cv2.circle(output2, (x, y), r, (0, 255, 0), 4)
	cv2.rectangle(output2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
plt.figure();plt.imshow(output2)


circles_num=np.zeros((96,3))
circles_num[:,2]=50
#d_vert=(np.max(circles_sorted1[1:,1])-np.min(circles_sorted1[1:,1]))/11
#d_horz=(np.max(circles_sorted1[1:,0])-np.min(circles_sorted1[1:,0]))/7
#init_horz=np.min(circles_sorted1[1:,0])
#init_vert=np.min(circles_sorted1[1:,1])
#horz_pos=np.min(circles_sorted1[1:,0])
#vert_pos=init_vert.copy()

d_vert=(np.array([255,1862.69])-np.array([230.258,296.212]))/11
d_horz=(np.array([1203.55,280.99])-np.array([230.258,296.212]))/7
init_pos=np.array([230.258,296.212])
pos=init_pos.copy()

for i in range(96):
    circles_num[i,[0,1]]=pos
    pos=pos+d_horz
    if np.mod(i,8)==7:
        pos=init_pos.copy()
        pos=pos+d_vert*np.ceil(i/8)        

circles_num=np.round(circles_num).astype(int)
output2 = ndimage.rotate(image, -ang, reshape=False)

for (x, y, r) in circles_num:
	# draw the circle in the output image, then draw a rectangle
	# corresponding to the center of the circle
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

#plt.figure();plt.imshow(mask)

#plt.figure();plt.imshow(image_HSV[:,:,0]);plt.colorbar()
#plt.figure();plt.imshow(image_HSV[:,:,1]);plt.colorbar()
#plt.figure();plt.imshow(image_HSV[:,:,2]);plt.colorbar()


intensity=np.zeros((96,3))
for i in range(circles_num.shape[0]):
    collect_int=np.zeros(4*r*r)
    collect_int[:]=np.nan
    l=0
    for j in range(circles_num[i,1]-r,circles_num[i,1]+r):
        for k in range(circles_num[i,0]-r,circles_num[i,0]+r):
            if (j-circles_num[i,1])**2 +(k-circles_num[i,0])**2 <= r*r:
                 collect_int[l]=image_HSV[j,k,1]
                 l=l+1
    print(np.mean(np.isnan(collect_int)))
    intensity[i,0]=circles_num[i,0];intensity[i,1]=circles_num[i,1];
    intensity[i,2]=np.nanmean(collect_int)

import pandas as pd
path_planilha=r'E:\Downloads_HD\AnaliseImagem-ELISA\EspectroFotometro'
planilha=pd.read_excel(path_planilha+'\\ELISA 230620.xlsx',sheet_name='Placa transparente',header=0)
ind1=np.argsort(intensity[:,0])
int_sorted1=intensity[ind1,:].copy()
int_norm1=np.round((intensity[:,0]-np.min(intensity[:,0]))/((np.max(intensity[:,0])-np.min(intensity[:,0]))/7))
int_norm2=np.round((intensity[:,1]-np.min(intensity[:,1]))/((np.max(intensity[:,1])-np.min(intensity[:,1]))/11))

DF_dados=pd.DataFrame(intensity)
DF_dados['absorb']=0

for i in range(intensity.shape[0]):
    DF_dados.iloc[i,3]=planilha.iloc[int_norm1[i].astype(int),int_norm2[i].astype(int)]
    
DF_dados.plot(x=2,y='absorb',style='+')
DF_dados['log absorb']=np.log(DF_dados['absorb'])
print(DF_dados.corr())
DF_dados.plot(x=2,y='log absorb',style='+')
